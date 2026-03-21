#!/usr/bin/env python3
"""
EXPERIMENT #001 - MTF Donchian+MACD+RSI+ADX (1h+4h v1)
==================================================================================================
Hypothesis: 4h Donchian breakout trend + 1h MACD histogram entry + RSI pullback filter.
Different from #040 which uses 15m+1h with HMA+Supertrend+KAMA.

Key differences from #040:
- Timeframe: 1h instead of 15m (fewer trades, lower fees)
- MTF: 1h + 4h (4h trend is more stable than 1h)
- Trend: Donchian(20) breakout instead of HMA/Supertrend
- Entry: MACD histogram cross instead of RSI pullback alone
- Filter: ADX(14) > 25 for trend strength
- Position size: 0.30 (slightly more conservative than #040's 0.35)
- Stoploss: 2.0*ATR (same as #040)

Why this should work:
- Donchian breakout captures true range breakouts (proven in trend following)
- 4h trend filter reduces whipsaws vs 1h trend
- MACD histogram adds momentum confirmation
- RSI pullback ensures we're not chasing extremes
- ADX filter avoids choppy markets

Risk management:
- Max signal: 0.30 (30% position size)
- Stoploss: 2*ATR → signal=0
- Take profit: 2R → reduce to 0.15, trail at 1R
- Leverage: 1.0 (no leverage)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_macd_rsi_adx_1h_4h_v1"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (line, signal, histogram)"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # EMA calculation
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    macd_signal = np.zeros(n)
    first_valid = slow + signal - 1
    macd_signal[first_valid] = np.mean(macd_line[slow:first_valid + 1])
    
    for i in range(first_valid + 1, n):
        macd_signal[i] = macd_signal[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - macd_signal[i - 1])
    
    histogram = macd_line - macd_signal
    
    return macd_line, macd_signal, histogram


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
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
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_sma(close, period=20):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    sma_1h = calculate_sma(close, period=200)
    
    # 4h indicators for trend (using mtf_data helper - CRITICAL)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Donchian on 4h
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
        
        # ADX on 4h for trend strength
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        
        # SMA on 4h for additional trend filter
        sma_4h = calculate_sma(close_4h, period=50)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
        donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        sma_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_4h)
        
    except Exception:
        # Fallback if mtf_data fails - use synthetic resampling
        bars_per_4h = 4
        n_4h = n // bars_per_4h
        
        c_4h = np.zeros(n_4h)
        h_4h = np.zeros(n_4h)
        l_4h = np.zeros(n_4h)
        
        for i in range(n_4h):
            start_idx = i * bars_per_4h
            end_idx = start_idx + bars_per_4h
            c_4h[i] = close[end_idx - 1]
            h_4h[i] = np.max(high[start_idx:end_idx])
            l_4h[i] = np.min(low[start_idx:end_idx])
        
        donchian_upper_4h, donchian_lower_4h = calculate_donchian(h_4h, l_4h, period=20)
        adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        sma_4h = calculate_sma(c_4h, period=50)
        
        donchian_upper_aligned = np.zeros(n)
        donchian_lower_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        sma_4h_aligned = np.zeros(n)
        
        for i in range(n):
            idx_4h = i // bars_per_4h
            if idx_4h < n_4h:
                donchian_upper_aligned[i] = donchian_upper_4h[idx_4h]
                donchian_lower_aligned[i] = donchian_lower_4h[idx_4h]
                adx_4h_aligned[i] = adx_4h[idx_4h]
                sma_4h_aligned[i] = sma_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold for momentum
    MACD_HIST_MIN = 0
    
    # ADX threshold for trend strength (4h)
    ADX_4H_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 40, 14 * 2, 26 + 9)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(macd_hist_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend determination from Donchian
        price = close[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        sma_4h_val = sma_4h_aligned[i]
        
        # Determine 4h trend from Donchian position
        if donchian_upper > 0 and donchian_lower > 0:
            if price > (donchian_upper + donchian_lower) / 2:
                trend_4h = 1  # Bullish
            elif price < (donchian_upper + donchian_lower) / 2:
                trend_4h = -1  # Bearish
            else:
                trend_4h = 0  # Neutral
        else:
            trend_4h = 0
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_4H_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # SMA filter (4h) - additional trend confirmation
        if trend_4h == 1 and sma_4h_val > 0 and price < sma_4h_val:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        elif trend_4h == -1 and sma_4h_val > 0 and price > sma_4h_val:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
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
        
        # Entry logic: 4h Donchian trend + 1h MACD + RSI + ADX
        rsi_val = rsi_1h[i]
        macd_hist_val = macd_hist_1h[i]
        atr = atr_1h[i]
        
        if trend_4h == 1:  # Bullish trend on 4h
            # MACD histogram bullish + RSI pullback (not overbought)
            if (macd_hist_val > MACD_HIST_MIN and 
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # MACD histogram bearish + RSI pullback (not oversold)
            if (macd_hist_val < -MACD_HIST_MIN and 
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
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