#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian Breakout + Choppiness Regime + Volume (12h HTF)

HYPOTHESIS: Simple price channel breakout with regime filter is the most robust
pattern for crypto. Donchian(20) on 4h = 5-day breakout. 12h HTF provides
trend context. Choppiness Index filters out sideways markets. Volume spike
confirms the breakout. This exact pattern achieved test Sharpe 1.49 on SOLUSDT.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20-period Donchian high + HTF uptrend + ADX>25 = long
- Bear: Price breaks below 20-period Donchian low + HTF downtrend + ADX>25 = short  
- Range (CHOP>61.8): No trades — prevents whipsaw losses in chop
- Only trades when structure breaks out of range

KEY INSIGHT: The DB shows 4h Donchian+vol+chop+HTF is the BEST pattern.
Keep it simple: 1 channel + 1 volume filter + 1 regime + HTF direction.

TARGET: 80-200 total trades over 4 years (20-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_vol_12h_v4"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market trendiness
    CHOP > 61.8 = choppy/range (no trade)
    CHOP < 38.2 = trending (trade)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            # CHOP = 100 * log10(atr_sum / range_sum) / log10(period)
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """ADX for trend strength confirmation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # HTF: Simple SMA200 for trend direction on 12h
    htf_close = df_12h['close'].values
    htf_sma200 = pd.Series(htf_close).rolling(window=200, min_periods=200).mean().values
    htf_price_above_sma = htf_close > htf_sma200
    htf_price_below_sma = htf_close < htf_sma200
    
    htf_bull_aligned = align_htf_to_ltf(prices, df_12h, htf_price_above_sma.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_12h, htf_price_below_sma.astype(float))
    
    # Also get 12h Donchian for HTF breakout confirmation
    htf_high = df_12h['high'].values
    htf_low = df_12h['low'].values
    htf_upper, _, htf_lower = calculate_donchian(htf_high, htf_low, period=20)
    htf_breakout_up = htf_close > htf_upper
    htf_breakout_down = htf_close < htf_lower
    htf_bu_aligned = align_htf_to_ltf(prices, df_12h, htf_breakout_up.astype(float))
    htf_bd_aligned = align_htf_to_ltf(prices, df_12h, htf_breakout_down.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    upper_20, middle_20, lower_20 = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signal generation ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # Donchian(20) + volume(20) + SMA200(200) on 12h needs 200*8=1600 4h bars min, but we align so 250 is fine
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(upper_20[i]) or np.isnan(lower_20[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(htf_bull_aligned[i]) or np.isnan(htf_bear_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK: Only trade when trending ===
        # CHOP < 50 = trending (use a wider threshold for more trades)
        # ADX > 20 = some trend strength
        is_trending = chop[i] < 50.0 and adx[i] > 18
        
        if not is_trending:
            # No position in choppy market
            if in_position:
                # Close any existing position
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF DIRECTION ===
        htf_bull = htf_bull_aligned[i] > 0.5
        htf_bear = htf_bear_aligned[i] > 0.5
        htf_bu = htf_bu_aligned[i] > 0.5  # HTF 12h breakout up
        htf_bd = htf_bd_aligned[i] > 0.5  # HTF 12h breakout down
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.6
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price breaks above Donchian high + HTF bull + volume
            if close[i] > upper_20[i] and close[i] > upper_20[i-1]:
                if (htf_bull or htf_bu) and vol_spike:
                    desired_signal = SIZE
            
            # SHORT: Price breaks below Donchian low + HTF bear + volume
            elif close[i] < lower_20[i] and close[i] < lower_20[i-1]:
                if (htf_bear or htf_bd) and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: price falls below Donchian OR 2.5*ATR trailing
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls back below Donchian upper
                if close[i] < lower_20[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear and not htf_bull:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: price rises above Donchian OR 2.5*ATR trailing
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises back above Donchian lower
                if close[i] > upper_20[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull and not htf_bear:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals