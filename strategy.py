#!/usr/bin/env python3
"""
EXPERIMENT #007 - MTF ADX+RSI+MACD Divergence (1h+4h+1d v1)
==================================================================================================
Hypothesis: Use 1h as primary timeframe (cleaner than 15m/30m, fewer whipsaws) with:
- 4h ADX + Supertrend for trend strength and direction
- 1d SMA(50/200) for regime filter (bull/bear market)
- 1h RSI divergence + MACD histogram for entry timing
- BBW on 1h for volatility regime

Why this should work:
- 1h timeframe has proven success in historical strategies (less noise than 15m/30m)
- ADX > 25 filters out choppy markets (reduces false signals)
- RSI divergence catches reversals early (vs simple RSI levels)
- Daily regime filter avoids counter-trend trades in strong trends
- Three timeframes provide robust confirmation without overfitting

Different from failed experiments:
- Primary=1h (not 15m/30m like #003, #004, #006)
- ADX filter (not used in recent failures)
- RSI divergence detection (not just RSI levels)
- Daily regime filter (1d SMA50/200)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_adx_rsi_div_macd_1h_4h_1d_v1"
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
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    plus_di_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    
    for i in range(n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_di_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_di_smooth[i] / tr_smooth[i]
        
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return plus_di, minus_di, adx


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period - 1] = lower_band[period - 1]
    
    for i in range(period, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    for i in range(n):
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def detect_rsi_divergence(rsi, close, lookback=5):
    """Detect bullish/bearish RSI divergence"""
    n = len(close)
    bullish_div = np.zeros(n)
    bearish_div = np.zeros(n)
    
    for i in range(lookback * 2, n):
        # Find local lows in price
        price_low_idx = None
        for j in range(i - lookback * 2, i - lookback):
            if j > 0 and close[j] <= close[j - 1] and close[j] <= close[j + 1]:
                price_low_idx = j
                break
        
        if price_low_idx is not None:
            # Check if RSI made higher low while price made lower low
            rsi_low_idx = None
            for j in range(i - lookback, i):
                if j > 0 and rsi[j] <= rsi[j - 1] and rsi[j] <= rsi[j + 1]:
                    rsi_low_idx = j
                    break
            
            if rsi_low_idx is not None and rsi_low_idx > price_low_idx:
                if close[i] < close[price_low_idx] and rsi[i] > rsi[rsi_low_idx]:
                    bullish_div[i] = 1
        
        # Find local highs in price
        price_high_idx = None
        for j in range(i - lookback * 2, i - lookback):
            if j > 0 and close[j] >= close[j - 1] and close[j] >= close[j + 1]:
                price_high_idx = j
                break
        
        if price_high_idx is not None:
            # Check if RSI made lower high while price made higher high
            rsi_high_idx = None
            for j in range(i - lookback, i):
                if j > 0 and rsi[j] >= rsi[j - 1] and rsi[j] >= rsi[j + 1]:
                    rsi_high_idx = j
                    break
            
            if rsi_high_idx is not None and rsi_high_idx > price_high_idx:
                if close[i] > close[price_high_idx] and rsi[i] < rsi[rsi_high_idx]:
                    bearish_div[i] = 1
    
    return bullish_div, bearish_div


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    _, _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    plus_di_1h, minus_di_1h, adx_1h = calculate_adx(high, low, close, period=14)
    bullish_div_1h, bearish_div_1h = detect_rsi_divergence(rsi_1h, close, lookback=5)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h indicators
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        _, _, adx_4h = calculate_adx(h_4h, l_4h, c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    except Exception:
        st_direction_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # Daily SMA50 and SMA200 for regime
        sma50_1d = pd.Series(c_1d).rolling(window=50, min_periods=50).mean().values
        sma200_1d = pd.Series(c_1d).rolling(window=200, min_periods=200).mean().values
        
        # Align to 1h timeframe
        sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
        sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
        
        # Calculate regime: 1 = bull, -1 = bear, 0 = neutral
        regime_1d = np.zeros(n)
        for i in range(n):
            if i < len(sma50_1d_aligned) and i < len(sma200_1d_aligned):
                if sma50_1d_aligned[i] > sma200_1d_aligned[i] and c_1d is not None:
                    regime_1d[i] = 1
                elif sma50_1d_aligned[i] < sma200_1d_aligned[i]:
                    regime_1d[i] = -1
    except Exception:
        sma50_1d_aligned = np.zeros(n)
        sma200_1d_aligned = np.zeros(n)
        regime_1d = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # ADX threshold for trend strength
    ADX_MIN = 25
    
    # RSI thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # BBW minimum for regime filter
    BBW_MIN = 0.02
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(250, 14 * 3, 50, 200)
    
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
        
        # Get aligned MTF values
        st_trend_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        adx_4h = adx_4h_aligned[i] if i < len(adx_4h_aligned) else 0
        daily_regime = regime_1d[i] if i < len(regime_1d) else 0
        
        # BBW filter - avoid choppy markets (1h)
        if bbw_1h[i] < BBW_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # 4h ADX filter - need strong trend
        if adx_4h < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
        # Daily regime filter - only trade with daily trend
        if daily_regime == 0:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
            else:
                position_side[i] = 0
            continue
        
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
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
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
        
        # Entry logic: 4h trend + 4h ADX + 1h RSI/MACD + Daily regime
        price = close[i]
        
        # Long entry conditions
        if (st_trend_4h == 1 and adx_4h > ADX_MIN and daily_regime == 1 and
            bbw_1h[i] >= BBW_MIN):
            # Multiple entry triggers: RSI pullback OR bullish divergence OR MACD cross
            rsi_ok = RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX
            div_ok = bullish_div_1h[i] == 1
            macd_ok = macd_hist_1h[i] > 0 and (i > 0 and macd_hist_1h[i - 1] <= 0)
            
            if rsi_ok or div_ok or macd_ok:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        # Short entry conditions
        elif (st_trend_4h == -1 and adx_4h > ADX_MIN and daily_regime == -1 and
              bbw_1h[i] >= BBW_MIN):
            # Multiple entry triggers: RSI pullback OR bearish divergence OR MACD cross
            rsi_ok = RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX
            div_ok = bearish_div_1h[i] == 1
            macd_ok = macd_hist_1h[i] < 0 and (i > 0 and macd_hist_1h[i - 1] >= 0)
            
            if rsi_ok or div_ok or macd_ok:
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