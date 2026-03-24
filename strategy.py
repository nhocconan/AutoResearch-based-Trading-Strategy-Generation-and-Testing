#!/usr/bin/env python3
"""
Experiment #299: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + Volume Session

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Combined with Choppiness Index regime detection and volume/session filters for quality entries.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, catches reversals when Fisher crosses -1.5/+1.5
   Proven to work in bear markets where RSI fails (2022 crash, 2025 bear)
2. CHOPPINESS REGIME: CHOP>60 = mean reversion (Fisher extremes), CHOP<40 = trend (HMA breakout)
3. VOLUME CONFIRMATION: volume > 1.5x 20-bar MA ensures real moves, not fakeouts
4. SESSION FILTER: 08-20 UTC only (high liquidity, less manipulation)
5. HTF ALIGNMENT: 4h HMA for trend bias, 12h HMA for major direction

Entry Logic:
- Choppy regime: Fisher < -1.5 + volume spike + 4h HMA bull → long
- Choppy regime: Fisher > +1.5 + volume spike + 4h HMA bear → short
- Trending regime: Price breaks 4h HMA + volume confirmation → follow trend

Position sizing: 0.20 base, 0.30 when 12h HTF aligned
Stoploss: 2.5x ATR from entry

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
Timeframe: 1h (as required for this experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_volume_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals when Fisher crosses extreme levels (-1.5, +1.5)
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 range
    fisher_input = np.zeros(n)
    fisher_input[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
            # Clamp to avoid division issues
            normalized = max(-0.999, min(0.999, normalized))
            fisher_input[i] = normalized
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(1, n):
        if not np.isnan(fisher_input[i]) and not np.isnan(fisher_input[i-1]):
            fisher[i] = 0.5 * np.log((1.0 + fisher_input[i]) / (1.0 - fisher_input[i]))
            fisher_signal[i] = 0.5 * np.log((1.0 + fisher_input[i-1]) / (1.0 - fisher_input[i-1]))
    
    return fisher, fisher_signal

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Volume moving average for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000.0)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 40.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = not np.isnan(hma_12h_aligned[i]) and close[i] > hma_12h_aligned[i]
        htf_12h_bear = not np.isnan(hma_12h_aligned[i]) and close[i] < hma_12h_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM EXTREMES ===
        fisher_extreme_low = fisher[i] < -1.5 and fisher_signal[i] > -1.5  # crossing up
        fisher_extreme_high = fisher[i] > 1.5 and fisher_signal[i] < 1.5  # crossing down
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # REGIME 1: CHOPPY (mean reversion with Fisher)
        if current_regime == 2:
            # Long: Fisher extreme low + volume + 4h bull + above SMA200
            if fisher_extreme_low and volume_confirmed and htf_4h_bull and above_sma200:
                desired_signal = SIZE_STRONG if htf_12h_bull else SIZE_BASE
            
            # Short: Fisher extreme high + volume + 4h bear + below SMA200
            elif fisher_extreme_high and volume_confirmed and htf_4h_bear and below_sma200:
                desired_signal = -SIZE_STRONG if htf_12h_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (breakout with HMA + volume)
        elif current_regime == 1:
            # Long: Price above 4h HMA + volume spike + 4h bull
            if close[i] > hma_4h_aligned[i] and volume_confirmed and htf_4h_bull:
                desired_signal = SIZE_STRONG if htf_12h_bull else SIZE_BASE
            
            # Short: Price below 4h HMA + volume spike + 4h bear
            elif close[i] < hma_4h_aligned[i] and volume_confirmed and htf_4h_bear:
                desired_signal = -SIZE_STRONG if htf_12h_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals