#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "PSAR BBPT ZLSMA BTC 1min"
timeframe = "1m"

def _psar(high, low, close, start=0.05, increment=0.05, maximum=0.13):
    """Calculate Parabolic SAR."""
    n = len(close)
    psar = np.zeros(n)
    trend = np.ones(n)
    af = np.zeros(n)
    
    psar[0] = low[0]
    trend[0] = 1
    af[0] = start
    
    ep = high[0]
    
    for i in range(1, n):
        psar[i] = psar[i-1] + af[i-1] * (ep - psar[i-1])
        
        if trend[i-1] == 1:
            if low[i] > psar[i]:
                psar[i] = min(psar[i], low[i-1], low[i] if i > 1 else low[i])
                if high[i] > ep:
                    ep = high[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    af[i] = af[i-1]
                trend[i] = 1
            else:
                trend[i] = -1
                psar[i] = ep
                af[i] = start
                ep = low[i]
        else:
            if high[i] < psar[i]:
                psar[i] = max(psar[i], high[i-1], high[i] if i > 1 else high[i])
                if low[i] < ep:
                    ep = low[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    af[i] = af[i-1]
                trend[i] = -1
            else:
                trend[i] = 1
                psar[i] = ep
                af[i] = start
                ep = high[i]
    
    return psar, trend

def _linreg(series, length, offset=0):
    """Calculate linear regression."""
    n = len(series)
    result = np.full(n, np.nan)
    
    for i in range(length - 1, n):
        x = np.arange(length)
        y = series[i - length + 1:i + 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            result[i] = slope * (length - 1 + offset) + intercept
    
    return result

def _zlsma(close, length, offset=0):
    """Calculate Zero Lag SMA."""
    lsma = _linreg(close, length, offset)
    lsma2 = _linreg(lsma, length, offset)
    eq = lsma - lsma2
    zlsma = lsma + eq
    return zlsma

def _atr(high, low, close, length=14):
    """Calculate Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
    
    return atr

def _highest(series, length):
    """Calculate highest value over lookback."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        result[i] = np.max(series[i - length + 1:i + 1])
    return result

def _lowest(series, length):
    """Calculate lowest value over lookback."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        result[i] = np.min(series[i - length + 1:i + 1])
    return result

def _ema(series, length):
    """Calculate Exponential Moving Average."""
    n = len(series)
    result = np.zeros(n)
    multiplier = 2 / (length + 1)
    result[0] = series[0]
    for i in range(1, n):
        result[i] = (series[i] - result[i-1]) * multiplier + result[i-1]
    return result

def _stochastic(high, low, close, k_length=14, k_smooth=3, d_smooth=3):
    """Calculate Stochastic oscillator."""
    n = len(close)
    lowest_low = _lowest(low, k_length)
    highest_high = _highest(high, k_length)
    
    stoch_k_raw = np.zeros(n)
    for i in range(n):
        if highest_high[i] != lowest_low[i]:
            stoch_k_raw[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
    
    stoch_k = np.zeros(n)
    for i in range(k_smooth - 1, n):
        stoch_k[i] = np.mean(stoch_k_raw[i - k_smooth + 1:i + 1])
    
    stoch_d = np.zeros(n)
    for i in range(d_smooth - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_smooth + 1:i + 1])
    
    return stoch_k, stoch_d

def _bollinger_bands(close, length=20, std_mult=2):
    """Calculate Bollinger Bands."""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def _is_in_session(timestamps, session_start_hour, session_end_hour, tz_offset=0):
    """Check if timestamp is within session hours (UTC)."""
    hours = timestamps.dt.hour + tz_offset
    hours = hours % 24
    
    if session_start_hour < session_end_hour:
        in_session = (hours >= session_start_hour) & (hours < session_end_hour)
    else:
        in_session = (hours >= session_start_hour) | (hours < session_end_hour)
    
    return in_session

def generate_signals(prices):
    """Generate target position signals based on PSAR, BBPT, ZLSMA strategy."""
    n = len(prices)
    if n == 0:
        return np.array([])
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    timestamps = pd.to_datetime(prices['open_time'])
    
    psar, psar_dir = _psar(high, low, close)
    
    zlsma = _zlsma(close, 50, 0)
    
    atr5 = _atr(high, low, close, 5)
    highest_50 = _highest(high, 50)
    lowest_50 = _lowest(low, 50)
    
    bull_trend = np.zeros(n)
    bear_trend = np.zeros(n)
    for i in range(n):
        if atr5[i] != 0:
            bull_trend[i] = (close[i] - lowest_50[i]) / atr5[i]
            bear_trend[i] = (highest_50[i] - close[i]) / atr5[i]
    
    bear_trend2 = -1 * bear_trend
    trend = bull_trend - bear_trend
    
    bull_trend_hist = np.zeros(n)
    bear_trend_hist = np.zeros(n)
    for i in range(n):
        if bull_trend[i] < 2:
            bull_trend_hist[i] = bull_trend[i] - 2
        if bear_trend2[i] > -2:
            bear_trend_hist[i] = bear_trend2[i] + 2
    
    ema50 = _ema(close, 50)
    
    stoch_k, stoch_d = _stochastic(high, low, close)
    bb_middle, bb_upper, bb_lower = _bollinger_bands(close)
    
    psar_buy = np.zeros(n, dtype=bool)
    psar_sell = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if psar_dir[i] == 1 and psar_dir[i-1] == -1:
            psar_buy[i] = True
        if psar_dir[i] == -1 and psar_dir[i-1] == 1:
            psar_sell[i] = True
    
    zlsma_buy = np.zeros(n, dtype=bool)
    zlsma_sell = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(zlsma[i]):
            zlsma_buy[i] = (close[i] > zlsma[i]) and (open_price[i] > zlsma[i]) and (low[i] > zlsma[i]) and (high[i] > zlsma[i])
            zlsma_sell[i] = (close[i] < zlsma[i]) and (open_price[i] < zlsma[i]) and (low[i] < zlsma[i]) and (high[i] < zlsma[i])
    
    bbpt_buy = bear_trend_hist > 0
    bbpt_sell = bull_trend_hist > 0
    
    ema_buy = close > ema50
    ema_sell = ema50 > close
    
    zlsma_up = np.zeros(n, dtype=bool)
    zlsma_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(zlsma[i]) and not np.isnan(zlsma[i-1]):
            zlsma_up[i] = (zlsma[i] - zlsma[i-1]) > 1
            zlsma_down[i] = (zlsma[i] - zlsma[i-1]) < -1
    zlsma_up[0] = True
    zlsma_down[0] = True
    
    sl_check = np.zeros(n)
    for i in range(n):
        if close[i] != 0 and not np.isnan(zlsma[i]):
            sl_check[i] = (abs(close[i] - zlsma[i]) / close[i] * 100) + 0.02
    sl_ok = sl_check <= 0.2
    
    london_session = _is_in_session(timestamps, 8, 17, 0)
    ny_session = _is_in_session(timestamps, 13, 22, 0)
    tokyo_session = _is_in_session(timestamps, 0, 9, 0)
    sydney_session = _is_in_session(timestamps, 22, 7, 0)
    in_session = london_session | ny_session | tokyo_session | sydney_session
    
    long_condition = psar_buy & zlsma_buy & bbpt_buy & ema_buy & zlsma_up & sl_ok & in_session
    short_condition = psar_sell & zlsma_sell & bbpt_sell & ema_sell & zlsma_down & sl_ok & in_session
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    sl_long = 0.0
    sl_short = 0.0
    tp1_long = 0.0
    tp2_long = 0.0
    tp1_short = 0.0
    tp2_short = 0.0
    
    max_sl_pct = 0.2
    zlsma_offset = 0.02
    tp1_multi = 1.0
    tp2_multi = 2.0
    tp1_percent = 0.001
    
    for i in range(n):
        if position == 0:
            if long_condition[i] and close[i] != 0 and not np.isnan(zlsma[i]):
                sl_pct = ((close[i] - zlsma[i]) / close[i] * 100) + zlsma_offset
                tp1_pct = sl_pct * tp1_multi
                tp2_pct = sl_pct * tp2_multi
                
                entry_price = close[i]
                sl_long = entry_price * (1 - sl_pct / 100)
                tp1_long = entry_price * (1 + tp1_pct / 100)
                tp2_long = entry_price * (1 + tp2_pct / 100)
                position = 1
                signals[i] = 1
            
            elif short_condition[i] and close[i] != 0 and not np.isnan(zlsma[i]):
                sl_pct = ((zlsma[i] - close[i]) / close[i] * 100) + zlsma_offset
                tp1_pct = sl_pct * tp1_multi
                tp2_pct = sl_pct * tp2_multi
                
                entry_price = close[i]
                sl_short = entry_price * (1 + sl_pct / 100)
                tp1_short = entry_price * (1 - tp1_pct / 100)
                tp2_short = entry_price * (1 - tp2_pct / 100)
                position = -1
                signals[i] = -1
        
        elif position == 1:
            if i > 0:
                sl_long = max(sl_long, entry_price)
            
            if low[i] <= sl_long:
                position = 0
                signals[i] = 0
            elif high[i] >= tp1_long:
                if high[i] >= tp2_long:
                    position = 0
                    signals[i] = 0
                else:
                    position = 0
                    signals[i] = 0
        
        elif position == -1:
            if i > 0:
                sl_short = min(sl_short, entry_price)
            
            if high[i] >= sl_short:
                position = 0
                signals[i] = 0
            elif low[i] <= tp1_short:
                if low[i] <= tp2_short:
                    position = 0
                    signals[i] = 0
                else:
                    position = 0
                    signals[i] = 0
        
        else:
            signals[i] = 0
    
    return signals.astype(np.float64)