#!/usr/bin/env python3
"""
EXPERIMENT #038 - MTF HMA+RSI+ADX 1h+4h v1
==================================================================================================
Hypothesis: Simplify the winning 30m+4h approach to 1h+4h with ADX trend strength filter.
Current best (mtf_hma_rsi_30m_4h_simplified_v2) has Sharpe=1.153. This strategy:
- Uses 1h base (more trades than 4h, cleaner than 15m/30m)
- 4h HMA for trend direction + 4h ADX for trend strength (avoid choppy 4h markets)
- 1h RSI for pullback entries (proven in #027, #036)
- Simpler logic than current #038 (no MACD, no BBW, no Supertrend complexity)
- Position size: 0.30 (conservative for drawdown control)

Why this should beat current best:
- 1h generates ~2x trades vs 4h while maintaining signal quality
- ADX(14)>25 on 4h filters out weak/choppy trends (reduces whipsaws)
- Simpler entry logic = fewer false signals, lower churn costs
- Proven HMA+RSI combination from successful strategies #027, #036
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_adx_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth +DM, -DM, and TR using Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    atr = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_smooth = np.sum(plus_dm[1:period])
    minus_smooth = np.sum(minus_dm[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_smooth = plus_smooth - plus_smooth / period + plus_dm[i]
        minus_smooth = minus_smooth - minus_smooth / period + minus_dm[i]
        
        if atr[i] > 0:
            plus_di[i] = 100 * plus_smooth / atr[i]
            minus_di[i] = 100 * minus_smooth / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators for trend filter
        hma_4h = calculate_hma(c_4h, period=21)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # ADX threshold for trend strength (avoid choppy markets)
    ADX_MIN = 25
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 21 * 2, 14 * 3)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 4h values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        adx_4h_val = adx_4h_aligned[i] if i < len(adx_4h_aligned) else 0
        
        # Get 4h price for trend direction
        try:
            df_4h = get_htf_data(prices, '4h')
            c_4h = df_4h['close'].values
            idx_4h = min(i // 4, len(c_4h) - 1)  # 4 x 1h = 4h
            price_4h = c_4h[idx_4h] if idx_4h >= 0 and idx_4h < len(c_4h) else 0
        except Exception:
            price_4h = close[i]
        
        # 4h ADX filter - only trade when trend is strong
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # 4h trend direction (price vs HMA)
        trend_4h = 0
        if price_4h > hma_4h_val and hma_4h_val > 0:
            trend_4h = 1
        elif price_4h < hma_4h_val and hma_4h_val > 0:
            trend_4h = -1
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + ADX strength + 1h RSI pullback
        price = close[i]
        
        if trend_4h == 1:  # Bullish trend on 4h
            # RSI pullback on 1h (not overbought)
            if RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # RSI pullback on 1h (not oversold)
            if RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals