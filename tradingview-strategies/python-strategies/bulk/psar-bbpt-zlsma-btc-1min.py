#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "PSAR BBPT ZLSMA BTC 1min"
timeframe = "1m"
leverage = 1

def calculate_psar(high, low, close, start=0.05, increment=0.05, maximum=0.13):
    n = len(close)
    psar = np.zeros(n)
    direction = np.ones(n)
    af = np.zeros(n)
    
    psar[0] = low[0]
    direction[0] = 1
    af[0] = start
    
    for i in range(1, n):
        if direction[i-1] == 1:
            psar[i] = psar[i-1] + af[i-1] * (high[i-1] - psar[i-1])
            if low[i] < psar[i]:
                direction[i] = -1
                psar[i] = high[i-1]
                af[i] = start
            else:
                direction[i] = 1
                if high[i] > high[i-1]:
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    af[i] = af[i-1]
        else:
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - low[i-1])
            if high[i] > psar[i]:
                direction[i] = 1
                psar[i] = low[i-1]
                af[i] = start
            else:
                direction[i] = -1
                if low[i] < low[i-1]:
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    af[i] = af[i-1]
    
    return psar, direction

def calculate_linreg(src, length, offset=0):
    n = len(src)
    result = np.full(n, np.nan)
    
    for i in range(length - 1, n):
        x = np.arange(length)
        y = src[i - length + 1:i + 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            result[i] = slope * (length - 1 + offset) + intercept
    
    return result

def calculate_zlsma(close, length=50, offset=0):
    lsma = calculate_linreg(close, length, offset)
    lsma2 = calculate_linreg(lsma, length, offset)
    eq = lsma - lsma2
    zlsma = lsma + eq
    return zlsma

def calculate_atr(high, low, close, length=5):
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

def calculate_lowest(low, length):
    n = len(low)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        result[i] = np.min(low[i - length + 1:i + 1])
    return result

def calculate_highest(high, length):
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(length - 1, n):
        result[i] = np.max(high[i - length + 1:i + 1])
    return result

def calculate_ema(close, length=50):
    n = len(close)
    ema = np.zeros(n)
    multiplier = 2 / (length + 1)
    ema[0] = close[0]
    
    for i in range(1, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    
    if n == 0:
        return np.array([], dtype=np.int8)
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values
    
    psar, psar_dir = calculate_psar(high, low, close, 0.05, 0.05, 0.13)
    zlsma = calculate_zlsma(close, 50, 0)
    
    atr = calculate_atr(high, low, close, 5)
    lowest_50 = calculate_lowest(low, 50)
    highest_50 = calculate_highest(high, 50)
    
    bull_trend = np.zeros(n)
    bear_trend = np.zeros(n)
    for i in range(n):
        if not np.isnan(atr[i]) and atr[i] != 0 and not np.isnan(lowest_50[i]) and not np.isnan(highest_50[i]):
            bull_trend[i] = (close[i] - lowest_50[i]) / atr[i]
            bear_trend[i] = (highest_50[i] - close[i]) / atr[i]
    
    bull_trend_hist = np.zeros(n)
    bear_trend_hist = np.zeros(n)
    for i in range(n):
        if bull_trend[i] < 2:
            bull_trend_hist[i] = bull_trend[i] - 2
        if -bear_trend[i] > -2:
            bear_trend_hist[i] = -bear_trend[i] + 2
    
    bbpt_buy = bear_trend_hist
    bbpt_sell = bull_trend_hist
    
    ema = calculate_ema(close, 50)
    
    zlsma_diff = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(zlsma[i]) and not np.isnan(zlsma[i-1]):
            zlsma_diff[i] = zlsma[i] - zlsma[i-1]
    
    signals = np.zeros(n, dtype=np.int8)
    
    in_long = False
    in_short = False
    long_sl = 0.0
    short_sl = 0.0
    long_tp1 = 0.0
    long_tp2 = 0.0
    short_tp1 = 0.0
    short_tp2 = 0.0
    entry_price = 0.0
    
    max_sl = 0.2
    zlsma_offset = 0.02
    tp1_multi = 1.0
    tp2_multi = 2.0
    
    ema_filter = False
    zlsma_angle_filter = True
    
    for i in range(n):
        psar_buy = psar_dir[i] == 1 and i > 0 and psar_dir[i-1] == -1
        psar_sell = psar_dir[i] == -1 and i > 0 and psar_dir[i-1] == 1
        
        zlsma_buy = (not np.isnan(zlsma[i]) and close[i] > zlsma[i] and 
                     open_price[i] > zlsma[i] and low[i] > zlsma[i] and high[i] > zlsma[i])
        zlsma_sell = (not np.isnan(zlsma[i]) and close[i] < zlsma[i] and 
                      open_price[i] < zlsma[i] and low[i] < zlsma[i] and high[i] < zlsma[i])
        
        bbpt_buy_cond = bbpt_buy[i] < 0
        bbpt_sell_cond = bbpt_sell[i] < 0
        
        ema_buy = True
        ema_sell = True
        if ema_filter:
            ema_buy = not np.isnan(ema[i]) and close[i] > ema[i]
            ema_sell = not np.isnan(ema[i]) and ema[i] > close[i]
        
        zlsma_up = True
        zlsma_down = True
        if zlsma_angle_filter and i > 0:
            zlsma_up = zlsma_diff[i] > 1
            zlsma_down = zlsma_diff[i] < -1
        
        sl_check = 0.0
        if not np.isnan(zlsma[i]) and close[i] != 0:
            sl_check = (abs(close[i] - zlsma[i]) / close[i] * 100) + zlsma_offset
        sl_ok = sl_check <= max_sl
        
        long_condition = (psar_buy and zlsma_buy and bbpt_buy_cond and 
                          ema_buy and zlsma_up and sl_ok)
        short_condition = (psar_sell and zlsma_sell and bbpt_sell_cond and 
                           ema_sell and zlsma_down and sl_ok)
        
        if not in_long and not in_short:
            if long_condition:
                signals[i] = 1
                in_long = True
                entry_price = close[i]
                if not np.isnan(zlsma[i]) and close[i] != 0:
                    sl_pct = ((close[i] - zlsma[i]) / close[i] * 100) + zlsma_offset
                    long_sl = close[i] * (1 - sl_pct / 100)
                    long_tp1 = entry_price * (1 + sl_pct * tp1_multi / 100)
                    long_tp2 = entry_price * (1 + sl_pct * tp2_multi / 100)
            elif short_condition:
                signals[i] = -1
                in_short = True
                entry_price = close[i]
                if not np.isnan(zlsma[i]) and close[i] != 0:
                    sl_pct = ((zlsma[i] - close[i]) / close[i] * 100) + zlsma_offset
                    short_sl = close[i] * (1 + sl_pct / 100)
                    short_tp1 = entry_price * (1 - sl_pct * tp1_multi / 100)
                    short_tp2 = entry_price * (1 - sl_pct * tp2_multi / 100)
        
        if in_long:
            if low[i] <= long_sl:
                signals[i] = -1
                in_long = False
                long_sl = 0.0
                long_tp1 = 0.0
                long_tp2 = 0.0
            elif long_tp1 > 0 and high[i] >= long_tp1:
                signals[i] = 0
                long_tp1 = 0.0
            elif long_tp2 > 0 and high[i] >= long_tp2:
                signals[i] = -1
                in_long = False
                long_sl = 0.0
                long_tp1 = 0.0
                long_tp2 = 0.0
        
        if in_short:
            if high[i] >= short_sl:
                signals[i] = 1
                in_short = False
                short_sl = 0.0
                short_tp1 = 0.0
                short_tp2 = 0.0
            elif short_tp1 > 0 and low[i] <= short_tp1:
                signals[i] = 0
                short_tp1 = 0.0
            elif short_tp2 > 0 and low[i] <= short_tp2:
                signals[i] = 1
                in_short = False
                short_sl = 0.0
                short_tp1 = 0.0
                short_tp2 = 0.0
    
    return signals