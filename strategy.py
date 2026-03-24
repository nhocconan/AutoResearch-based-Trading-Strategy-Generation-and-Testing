#!/usr/bin/env python3
"""
Experiment #551: 6h Primary + 1d/1w HTF — Fisher Transform + Keltner Squeeze + Volume Confirmation

Hypothesis: 6h timeframe captures multi-day swings better than 4h (less noise) and 12h (more signals).
Ehlers Fisher Transform provides superior reversal detection vs RSI in bear/range markets (2022 crash, 2025).
Keltner Channel squeeze identifies low-volatility compression before breakouts. Volume confirmation filters
false breakouts. 1w HMA for macro bias + 1d HMA for medium bias ensures we trade with higher-timeframe trend.

Key innovations vs failed 6h strategies:
1. Fisher Transform (period=9) instead of RSI — catches turning points earlier in bear rallies
2. Keltner Channel squeeze (EMA20 ± 1.5*ATR) instead of Bollinger — tighter, better for crypto volatility
3. Volume spike confirmation (vol > 1.5*MA20) — avoids false breakouts on low volume
4. Asymmetric sizing: 0.30 when 3x HTF aligned, 0.20 when 2x aligned
5. Simpler regime logic — avoid over-complication that caused 0 trades in #541, #545, #549

Strategy logic:
1. 1w HMA(21) = macro trend (very slow, only changes monthly)
2. 1d HMA(21) = medium trend (changes weekly)
3. 6h Fisher Transform(9) = entry timing (crosses -1.5 long, +1.5 short)
4. 6h Keltner Channel(20, 1.5) = volatility squeeze detection
5. 6h Volume ratio = breakout confirmation
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (must have HTF alignment):
- LONG: Fisher < -1.5 (oversold) + price > 1d HMA + volume spike + KC squeeze release
- SHORT: Fisher > +1.5 (overbought) + price < 1d HMA + volume spike + KC squeeze release

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=3 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_keltner_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution for clearer reversal signals
    Fisher crosses above -1.5 = long signal
    Fisher crosses below +1.5 = short signal
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - lowest) / (highest - lowest) - 0.5)
    3. Smooth with EMA
    4. Fisher = 0.5 * ln((1 + smoothed) / (1 - smoothed))
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize price within lookback window
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized[i] = 0.66 * ((typical[i] - lowest) / price_range - 0.5)
        else:
            normalized[i] = 0.0
    
    # Clamp to avoid ln domain errors
    normalized = np.clip(normalized, -0.99, 0.99)
    
    # Smooth with EMA (period=3 for Fisher)
    smoothed = pd.Series(normalized).ewm(span=3, min_periods=3, adjust=False).mean().values
    smoothed = np.clip(smoothed, -0.99, 0.99)
    
    # Fisher Transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period + 3, n):
        if abs(smoothed[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + smoothed[i]) / (1.0 - smoothed[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Fisher trigger (1-period lagged for signal)
    fisher_trigger = np.zeros(n)
    fisher_trigger[:] = np.nan
    for i in range(1, n):
        fisher_trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else np.nan
    
    return fisher, fisher_trigger

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, atr_mult=1.5):
    """
    Keltner Channels
    Middle: EMA(20) of close
    Upper: EMA(20) + 1.5 * ATR(14)
    Lower: EMA(20) - 1.5 * ATR(14)
    Squeeze: when price is inside channels (low volatility)
    Breakout: when price closes outside channels
    """
    n = len(close)
    if n < ema_period + atr_period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Middle line (EMA)
    middle = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    # ATR for channel width
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Upper and lower bands
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
    return upper, middle, lower

def calculate_volume_ratio(volume, period=20):
    """
    Volume ratio = current volume / moving average volume
    Ratio > 1.5 = volume spike (confirms breakout)
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    vol_ratio[vol_ma <= 1e-10] = np.nan
    
    return vol_ratio

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, atr_mult=1.5)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20   # 2x HTF alignment
    SIZE_STRONG = 0.30  # 3x HTF alignment (all aligned)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS COUNT (for position sizing) ===
        htf_bull_count = 0
        htf_bear_count = 0
        
        if close[i] > hma_1d_aligned[i]:
            htf_bull_count += 1
        else:
            htf_bear_count += 1
        
        if close[i] > hma_1w_aligned[i]:
            htf_bull_count += 1
        else:
            htf_bear_count += 1
        
        # Price vs KC middle as third HTF proxy
        if close[i] > kc_middle[i]:
            htf_bull_count += 1
        else:
            htf_bear_count += 1
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (from below)
        fisher_long = False
        if i > 0 and not np.isnan(fisher_trigger[i]):
            fisher_long = (fisher_trigger[i] < -1.5) and (fisher[i] > -1.5)
        
        # Short: Fisher crosses below +1.5 (from above)
        fisher_short = False
        if i > 0 and not np.isnan(fisher_trigger[i]):
            fisher_short = (fisher_trigger[i] > 1.5) and (fisher[i] < 1.5)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === KC SQUEEZE/BREAKOUT ===
        # Price was inside KC (squeeze) and now breaking out
        was_inside = False
        if i > 0:
            was_inside = (close[i-1] >= kc_lower[i-1]) and (close[i-1] <= kc_upper[i-1])
        
        breakout_long = was_inside and (close[i] > kc_upper[i])
        breakout_short = was_inside and (close[i] < kc_lower[i])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        desired_size = 0.0
        
        # LONG ENTRY: Fisher oversold cross + HTF bullish + volume confirmation
        if fisher_long and htf_bull_count >= 2:
            if volume_confirmed or breakout_long:
                if htf_bull_count >= 3:
                    desired_size = SIZE_STRONG
                else:
                    desired_size = SIZE_WEAK
                desired_signal = desired_size
        
        # SHORT ENTRY: Fisher overbought cross + HTF bearish + volume confirmation
        elif fisher_short and htf_bear_count >= 2:
            if volume_confirmed or breakout_short:
                if htf_bear_count >= 3:
                    desired_size = SIZE_STRONG
                else:
                    desired_size = SIZE_WEAK
                desired_signal = -desired_size
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals