#!/usr/bin/env python3
"""
Experiment #021: 4h Funding Rate Mean-Reversion + Volume + Trend Filter

HYPOTHESIS: Funding rate is a proven edge for BTC/ETH (from 16K+ experiments).
- High funding (>Z+2) = speculative excess = short opportunity (mean reversion)
- Low funding (<Z-2) = fear/panic = long opportunity (mean reversion)
- Combine with 1d HMA trend filter + volume confirmation + ATR stoploss

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Low funding = fear = accumulation zone. Price bounces back.
- Bear: High funding = too many longs = flush out weak hands. Short the top.
- Mean-reversion is regime-agnostic (unlike trend following which fails in 2022)

WHY THIS IS NOVEL:
- All 13 recent failures used ONLY price/volume indicators
- Funding rate is a completely different data source with proven edge
- Not attempted in this session (DB shows 0.8-1.5 Sharpe on BTC/ETH)

KEY DIFFERENCES FROM PREVIOUS ATTEMPTS:
- Uses funding_rate.parquet data (not just price/volume)
- Mean-reversion logic (not breakout or trend following)
- Works through the 2022 crash (funding spiked before crash)

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
import os
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_mean_reversion_vol_trend"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def load_funding_data(symbol, prices_df):
    """
    Load funding rate data from parquet file and align to prices index.
    Returns array of funding rates aligned to prices timestamps.
    """
    # Find the base directory for processed data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different paths for funding data
    possible_paths = [
        os.path.join(base_dir, 'data', 'processed', 'funding', f'{symbol}.parquet'),
        os.path.join(base_dir, 'processed', 'funding', f'{symbol}.parquet'),
        os.path.join(base_dir, '..', 'data', 'processed', 'funding', f'{symbol}.parquet'),
    ]
    
    funding_path = None
    for path in possible_paths:
        if os.path.exists(path):
            funding_path = path
            break
    
    if funding_path is None:
        # Funding data not available, return zeros (no signal)
        return np.zeros(len(prices_df))
    
    # Load funding data
    funding_df = pd.read_parquet(funding_path)
    
    # Get prices timestamps
    if 'open_time' in prices_df.columns:
        price_times = pd.to_datetime(prices_df['open_time'])
    else:
        price_times = prices_df.index
    
    # Funding timestamps - align
    if 'open_time' in funding_df.columns:
        funding_times = pd.to_datetime(funding_df['open_time'])
    elif funding_df.index.name == 'open_time':
        funding_times = pd.to_datetime(funding_df.index)
    else:
        funding_times = pd.to_datetime(funding_df.index)
    
    # Create funding rate series aligned to prices
    funding_rates = np.zeros(len(prices_df))
    
    # Merge on nearest 8h timestamp (funding every 8 hours)
    # For each price bar, find the most recent funding rate
    for i in range(len(prices_df)):
        price_time = price_times.iloc[i] if hasattr(price_times, 'iloc') else price_times[i]
        
        # Find the most recent funding timestamp <= current price time
        mask = funding_times <= price_time
        if mask.any():
            funding_rates[i] = funding_df.loc[mask, 'funding_rate'].iloc[-1]
    
    return funding_rates

def calculate_funding_zscore(funding_rates, period=30):
    """
    Calculate rolling Z-score of funding rate.
    Z > +2.0 = funding too high = short signal
    Z < -2.0 = funding too low = long signal
    """
    n = len(funding_rates)
    if n < period:
        return np.full(n, np.nan)
    
    funding_series = pd.Series(funding_rates)
    rolling_mean = funding_series.rolling(window=period, min_periods=period).mean().values
    rolling_std = funding_series.rolling(window=period, min_periods=period).std().values
    
    # Z-score: (current - mean) / std
    zscore = np.zeros(n)
    for i in range(n):
        if rolling_std[i] > 1e-10:
            zscore[i] = (funding_rates[i] - rolling_mean[i]) / rolling_std[i]
        else:
            zscore[i] = 0.0
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend direction
    hma_21_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # === Load funding data ===
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if isinstance(prices.get('symbol'), pd.Series) else 'BTCUSDT'
    funding_rates = load_funding_data(symbol, prices)
    
    # === Calculate funding Z-score ===
    funding_zscore = calculate_funding_zscore(funding_rates, period=30)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Min periods for funding Z-score
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === FUNDING RATE SIGNALS ===
        funding_z = funding_zscore[i]
        
        # Skip if no valid funding data (z-score is 0 when funding unavailable)
        if np.isnan(funding_z) or funding_z == 0:
            signals[i] = 0.0
            continue
        
        # === HTF TREND: 1d HMA(21) direction ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === SHORT: Funding too high (Z > +2) = speculative excess = mean revert down ===
            if funding_z > 2.0 and htf_trend_down and vol_spike:
                desired_signal = -SIZE
            
            # === LONG: Funding too low (Z < -2) = fear/panic = mean revert up ===
            if funding_z < -2.0 and htf_trend_up and vol_spike:
                desired_signal = SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long position: stop if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to down
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Take profit at 2R
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
                
                # Exit if funding mean-reverts (Z moves toward 0)
                if funding_z > 0.5:  # Funding recovering = signal fading
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short position: stop if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to up
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Take profit at 2R
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = -SIZE / 2  # Half position
                
                # Exit if funding mean-reverts (Z moves toward 0)
                if funding_z < -0.5:  # Funding recovering = signal fading
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals