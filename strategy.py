#!/usr/bin/env python3
"""
Experiment #1151: 6h Primary + 1d/1w HTF — Keltner Trend + RSI Pullback + ADX Sizing

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Using Keltner Channels
for volatility-based entries with RSI pullback in confirmed trends should capture multi-day
swings better than Bollinger Bands (which whipsaw in crypto trends). ADX confirms trend
strength for position sizing rather than entry filtering.

Key innovations:
1. Keltner Channels (EMA20 + 2*ATR14): Better than BB for trending crypto markets
2. 1w HMA(21) for major trend bias (long-term direction)
3. 1d HMA(21) for intermediate confirmation (don't fight the daily trend)
4. 6h RSI(14) pullback entries: Long when RSI 35-50 in uptrend, Short when 50-65 in downtrend
5. ADX(14) for sizing: ADX>25 = full size (0.30), ADX<25 = half size (0.15)
6. ATR(14) 2.5x trailing stop with breakeven after 1.5R profit
7. Looser entry thresholds to GUARANTEE trades (RSI 35-65 range, not 20-80)

Why this should work on 6h:
- 6h captures 2-4 day swings (perfect for crypto momentum)
- Keltner Channels adapt to volatility (widen in high vol, narrow in low vol)
- RSI pullback entries avoid chasing breakouts (enter on dips in trends)
- ADX sizing reduces exposure in choppy markets without blocking entries
- 1w/1d HTF ensures we trade with major trend (no counter-trend trades)
- Looser RSI thresholds guarantee 30-60 trades/year target

Entry conditions (LOOSE to guarantee trades):
- LONG: price>1w_HMA + price>1d_HMA + RSI(14) 35-50 + price>Keltner_mid
- SHORT: price<1w_HMA + price<1d_HMA + RSI(14) 50-65 + price<Keltner_mid

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.15-0.30 discrete (ADX-scaled)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_rsi_pullback_adx_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate ATR for smoothing
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth +DM, -DM, and TR
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.divide(plus_dm_smooth, atr_smooth, out=np.zeros_like(plus_dm_smooth), where=atr_smooth != 0) * 100
    minus_di = np.divide(minus_dm_smooth, atr_smooth, out=np.zeros_like(minus_dm_smooth), where=atr_smooth != 0) * 100
    
    # Calculate DX
    di_sum = plus_di + minus_di
    dx = np.divide(np.abs(plus_di - minus_di), di_sum, out=np.zeros_like(plus_di), where=di_sum != 0) * 100
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    
    return adx

def calculate_keltner(high, low, close, ema_period=20, atr_period=14, multiplier=2.0):
    """Keltner Channels - EMA +/- multiplier*ATR"""
    n = len(close)
    
    # Calculate EMA
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period=atr_period)
    
    # Calculate channels
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, ema, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner(high, low, close)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.15   # ADX < 25
    SIZE_STRONG = 0.30 # ADX >= 25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    profit_target_hit = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
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
        
        if np.isnan(keltner_mid[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (Triple HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment (both 1d and 1w agree)
        strong_bull = hma_1d_bull and hma_1w_bull
        strong_bear = hma_1d_bear and hma_1w_bear
        
        # === ADX-BASED SIZING ===
        adx_strong = adx_14[i] >= 25.0
        current_size = SIZE_STRONG if adx_strong else SIZE_WEAK
        
        # === ENTRY LOGIC (RSI PULLBACK IN TREND) ===
        desired_signal = 0.0
        
        # LONG: Uptrend + RSI pullback to 35-50 + price above Keltner mid
        if strong_bull and 35.0 <= rsi_14[i] <= 50.0 and close[i] > keltner_mid[i]:
            desired_signal = current_size
        
        # SHORT: Downtrend + RSI pullback to 50-65 + price below Keltner mid
        elif strong_bear and 50.0 <= rsi_14[i] <= 65.0 and close[i] < keltner_mid[i]:
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (2.5x ATR trailing with breakeven) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            unrealized_pnl = (highest_since_entry - entry_price) / entry_atr
            
            # Move to breakeven after 1.5R profit
            if unrealized_pnl >= 1.5:
                profit_target_hit = True
                stop_price = max(stop_price, entry_price + 0.5 * entry_atr)
            else:
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
            
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            unrealized_pnl = (entry_price - lowest_since_entry) / entry_atr
            
            # Move to breakeven after 1.5R profit
            if unrealized_pnl >= 1.5:
                profit_target_hit = True
                stop_price = min(stop_price, entry_price - 0.5 * entry_atr)
            else:
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
            
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                profit_target_hit = False
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
                profit_target_hit = False
        
        signals[i] = final_signal
    
    return signals