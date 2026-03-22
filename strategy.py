#!/usr/bin/env python3
"""
Experiment #466: 4h Funding Rate Mean Reversion + Fisher Transform + HTF Trend

Hypothesis: After 465 failed experiments, the key insight is that perpetual futures
have a unique edge: funding rate mean reversion. When funding is extremely positive,
longs are overcrowded and price reverses. When extremely negative, shorts are
overcrowded and price squeezes higher. This is reported Sharpe 0.8-1.5 through 2022.

Strategy components:
1. FUNDING RATE Z-SCORE (primary signal): 
   - Load funding data from data/processed/funding/*.parquet
   - Z-score(30) of funding rate
   - Long when z < -2.0 (extreme negative funding = short squeeze)
   - Short when z > +2.0 (extreme positive funding = long liquidation)
   - This is the UNIQUE EDGE for perpetuals that spot strategies don't have

2. EHLERS FISHER TRANSFORM (entry timing):
   - Period=9, transforms price into Gaussian distribution
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Catches reversals better than RSI in bear markets

3. 1D HMA(21) TREND FILTER (via mtf_data helper):
   - Only long when price > 1d HMA (bull bias)
   - Only short when price < 1d HMA (bear bias)
   - HMA is smoother than EMA, critical for daily trend

4. VOL SPIKE CONFIRMATION:
   - ATR(7)/ATR(30) > 1.5 confirms real move (not noise)
   - Prevents entries during dead/choppy markets

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

6. POSITION SIZING: 0.30 discrete
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why this should work on 4h:
- Funding rate mean reversion is proven edge for perpetuals
- Fisher Transform catches reversals in bear/range markets
- 1D HMA prevents counter-trend trades
- Should work on BTC/ETH/SOL individually (funding exists for all)
- 4h timeframe captures funding cycle (8h funding, 4h captures 2 cycles/day)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_zscore_fisher_1d_hma_vol_spike_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - transforms price into Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Apply Fisher: 0.5 * ln((1 + x) / (1 - x))
    4. Smooth with EMA
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    typical = (high + low) / 2.0
    
    for i in range(period - 1, n):
        window_high = high[i-period+1:i+1].max()
        window_low = low[i-period+1:i+1].min()
        window_range = window_high - window_low
        
        if window_range < 1e-10:
            continue
        
        # Normalize to -1 to +1
        normalized = ((typical[i] - window_low) / window_range) * 2.0 - 1.0
        
        # Clamp to avoid log(0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with EMA of period 3
        if i == period - 1:
            fisher[i] = fisher_val
            fisher_signal[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_signal[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher_val
    
    return fisher, fisher_signal

def calculate_zscore(values, period=30):
    """Calculate Z-score of values relative to rolling mean."""
    values_s = pd.Series(values)
    rolling_mean = values_s.rolling(window=period, min_periods=period).mean()
    rolling_std = values_s.rolling(window=period, min_periods=period).std()
    zscore = (values_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def load_funding_data(prices):
    """
    Load funding rate data from processed parquet files.
    Returns funding rate array aligned with prices.
    """
    try:
        # Try to load funding data - path depends on symbol
        # The prices DataFrame should have symbol info or we infer from context
        # For now, try common paths
        import os
        import glob
        
        # Find funding parquet files
        funding_dir = "data/processed/funding"
        if os.path.exists(funding_dir):
            funding_files = glob.glob(f"{funding_dir}/*.parquet")
            if funding_files:
                # Load the first available funding file
                df_funding = pd.read_parquet(funding_files[0])
                
                # Funding data typically has: open_time, funding_rate
                # Need to align with prices open_time
                if 'open_time' in df_funding.columns and 'funding_rate' in df_funding.columns:
                    # Merge on open_time
                    prices_copy = prices.copy()
                    if 'open_time' in prices_copy.columns:
                        merged = prices_copy.merge(
                            df_funding[['open_time', 'funding_rate']],
                            on='open_time',
                            how='left'
                        )
                        # Forward fill missing funding rates
                        merged['funding_rate'] = merged['funding_rate'].ffill()
                        return merged['funding_rate'].values
        
        # Fallback: return zeros if no funding data available
        return np.zeros(len(prices))
    
    except Exception as e:
        # If funding data unavailable, return zeros
        return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    sma_50 = calculate_sma(close, 50)
    
    # Load funding rate data
    funding_rate = load_funding_data(prices)
    
    # Calculate funding Z-score
    funding_zscore = calculate_zscore(funding_rate, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            continue
        
        # === 1D HMA TREND FILTER ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOL SPIKE CONFIRMATION ===
        vol_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 1e-10 else 0
        vol_spike = vol_ratio > 1.5
        
        # === FUNDING RATE Z-SCORE SIGNAL ===
        # Extreme negative funding = long squeeze incoming = LONG
        # Extreme positive funding = short squeeze incoming = SHORT
        funding_long = funding_zscore[i] < -2.0
        funding_short = funding_zscore[i] > 2.0
        
        # === FISHER TRANSFORM ENTRY TIMING ===
        # Long when Fisher crosses above -1.5 (oversold reversal)
        # Short when Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher_signal[i] < -1.5 and fisher[i] > fisher_signal[i]
        fisher_short = fisher_signal[i] > 1.5 and fisher[i] < fisher_signal[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # FUNDING + FISHER + TREND alignment (all three must agree)
        # Long: extreme negative funding + fisher reversal + bull trend
        if funding_long and fisher_long and bull_trend_1d:
            new_signal = SIZE
        
        # Short: extreme positive funding + fisher reversal + bear trend
        elif funding_short and fisher_short and bear_trend_1d:
            new_signal = -SIZE
        
        # Fallback: Funding + Trend only (looser, ensures trades)
        elif funding_long and bull_trend_1d and vol_spike:
            new_signal = SIZE
        elif funding_short and bear_trend_1d and vol_spike:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals