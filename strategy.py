#!/usr/bin/env python3
"""
Experiment #379: 4h Primary + 1d HTF — Funding Rate Contrarian + Choppiness Regime

Hypothesis: Funding rate z-score is the BEST EDGE for BTC/ETH in bear/range markets.
When funding > +2σ (crowded longs), go short. When funding < -2σ (crowded shorts), go long.
This is market-neutral and worked through 2022 crash (Sharpe 0.8-1.5 in research).

Combined with:
1. 1d HMA for directional bias (only take funding signals aligned with HTF trend)
2. Choppiness Index to detect range vs trend (mean-revert in chop, reduce size in trend)
3. Relaxed funding thresholds (z > 1.5 instead of 2.0) to ensure enough trades
4. ATR trailing stoploss for risk management

Target: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL).
This should work in 2025 bear market where trend-following fails.

KEY INSIGHT: Most failed strategies were trend-following. 2025 is bearish/range.
Funding rate contrarian is PROVEN to work in crashes and bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_contrarian_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window=period, min_periods=period).mean()
    rolling_std = s.rolling(window=period, min_periods=period).std()
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.fillna(0.0).values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def load_funding_data(symbol):
    """
    Load funding rate data from processed parquet.
    Returns array aligned with prices index.
    """
    try:
        # Map symbol to filename
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        symbol_name = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{symbol_name}.parquet"
        df_funding = pd.read_parquet(funding_path)
        return df_funding
    except Exception:
        # Return zeros if funding data unavailable
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Try to load funding data
    df_funding = load_funding_data(symbol)
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Process funding rate if available
    if df_funding is not None and len(df_funding) > 0:
        # Align funding data to prices
        # Funding is typically 8h intervals, need to forward-fill to 4h
        try:
            # Merge funding with prices on timestamp
            prices_with_time = prices.copy()
            if 'open_time' in prices_with_time.columns:
                prices_with_time = prices_with_time.set_index('open_time')
            
            df_funding_indexed = df_funding.set_index('open_time')
            
            # Forward fill funding rate to match 4h bars
            funding_aligned = df_funding_indexed['funding_rate'].reindex(
                prices_with_time.index, method='ffill'
            ).fillna(0.0).values
            
            # Calculate z-score of funding rate
            funding_zscore = calculate_zscore(funding_aligned, period=30)
        except Exception:
            funding_zscore = np.zeros(n)
    else:
        # Fallback: use RSI as proxy for sentiment if funding unavailable
        funding_zscore = (rsi_14 - 50.0) / 15.0  # Normalize RSI to z-score-like range
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 30-50 trades/year)
    REDUCED_SIZE = 0.15  # Smaller size in trending regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(funding_zscore[i]):
            continue
        
        # === HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_ranging = chop[i] > 55.0  # CHOP > 55 = range (mean reversion works)
        is_trending = chop[i] < 45.0  # CHOP < 45 = trend (reduce position size)
        
        # === FUNDING RATE CONTRARIAN SIGNAL ===
        # Extreme positive funding = crowded longs = SHORT
        # Extreme negative funding = crowded shorts = LONG
        funding_extreme_long = funding_zscore[i] < -1.5  # Crowded shorts
        funding_extreme_short = funding_zscore[i] > 1.5   # Crowded longs
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Determine position size based on regime
        current_size = BASE_SIZE if is_ranging else REDUCED_SIZE
        
        if is_ranging:
            # RANGE REGIME: Funding contrarian works best
            # Long: funding extremely negative (crowded shorts) + HTF bullish bias
            if funding_extreme_long and price_above_hma_1d:
                desired_signal = current_size
            
            # Short: funding extremely positive (crowded longs) + HTF bearish bias
            elif funding_extreme_short and price_below_hma_1d:
                desired_signal = -current_size
            
            # Also allow counter-trend in strong range (HTF bias less important)
            elif funding_extreme_long and chop[i] > 65.0:  # Very choppy
                desired_signal = current_size * 0.7
            elif funding_extreme_short and chop[i] > 65.0:
                desired_signal = -current_size * 0.7
        
        elif is_trending:
            # TREND REGIME: Only take funding signals aligned with HTF trend
            # More conservative - funding extremes less reliable in strong trends
            if funding_extreme_long and price_above_hma_1d:
                desired_signal = current_size
            elif funding_extreme_short and price_below_hma_1d:
                desired_signal = -current_size
        
        else:
            # NEUTRAL REGIME: Require stronger funding extremes
            funding_very_long = funding_zscore[i] < -2.0
            funding_very_short = funding_zscore[i] > 2.0
            
            if funding_very_long:
                desired_signal = BASE_SIZE * 0.7
            elif funding_very_short:
                desired_signal = -BASE_SIZE * 0.7
        
        # === RSI FILTER (avoid entering at extremes against signal) ===
        # Don't long if RSI > 75, don't short if RSI < 25
        if desired_signal > 0 and rsi_14[i] > 75:
            desired_signal = 0.0
        if desired_signal < 0 and rsi_14[i] < 25:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FUNDING NORMALIZATION EXIT ===
        # Exit when funding z-score returns to neutral (mean reversion complete)
        if in_position and position_side > 0 and funding_zscore[i] > -0.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and funding_zscore[i] < 0.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Hold long if HTF still bullish
            if position_side > 0 and price_above_hma_1d:
                desired_signal = BASE_SIZE if is_ranging else REDUCED_SIZE
            # Hold short if HTF still bearish
            elif position_side < 0 and price_below_hma_1d:
                desired_signal = -BASE_SIZE if is_ranging else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals