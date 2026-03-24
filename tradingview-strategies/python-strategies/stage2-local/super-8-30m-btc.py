#!/usr/bin/env python3

name = "Super 8 - 30M BTC"
timeframe = "30m"
leverage = 10

import numpy as np
import pandas as pd


def generate_signals(prices):
    """
    Convert Pine Script Super 8 strategy to Python.
    Returns target position signals: 1 (long), -1 (short), 0 (flat)
    Uses next-bar execution semantics (no same-bar fills).
    """
    n = len(prices)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    
    signals = np.zeros(n, dtype=np.int8)
    
    # Extract OHLCV data
    close = prices['close'].values.astype(np.float64)
    high = prices['high'].values.astype(np.float64)
    low = prices['low'].values.astype(np.float64)
    open_price = prices['open'].values.astype(np.float64)
    volume = prices['volume'].values.astype(np.float64)
    
    # Input parameters (from Pine Script defaults)
    sEma_Length = 500
    fEma_Length = 100
    ADX_len = 28
    ADX_smo = 9
    th = 20.5
    Sst = 0.1
    Sinc = 0.04
    Smax = 0.4
    fastLength = 24
    slowLength = 52
    signalLength = 11
    lengthz = 14
    lengthStdev = 14
    A = -0.2
    B = 0.4
    volume_f = 0.4
    sma_Length = 55
    BB_Length = 40
    BB_mult = 2.2
    bbMinWidth01 = 5.0
    bbMinWidth02 = 2.0
    tp = 1.8
    trailOffset = 0.3
    DClength = 55
    sl = 8.0
    atrPeriodSl = 14
    multiplierPeriodSl = 15
    Pyr = 3
    bbBetterPrice = 0.7
    
    # Calculate indicators
    sEMA = ema(close, sEma_Length)
    fEMA = ema(close, fEma_Length)
    DIPlus, DIMinus, ADX = dmi(close, high, low, ADX_len, ADX_smo)
    SAR = parabolic_sar(high, low, Sst, Sinc, Smax)
    lMACD, sMACD_line, hist = macd(close, fastLength, slowLength, signalLength)
    
    # MAC-Z calculation
    zscore = calc_zvwap(close, volume, lengthz)
    macz = np.zeros_like(close)
    stdev_close = stdev(close, lengthStdev)
    for i in range(n):
        if not np.isnan(zscore[i]) and not np.isnan(lMACD[i]) and stdev_close[i] > 0:
            macz[i] = (zscore[i] * A) + (lMACD[i] / (stdev_close[i] * B))
        else:
            macz[i] = np.nan
    signal_macz = sma(macz, signalLength)
    histmacz = macz - signal_macz
    
    # Bollinger Bands
    BB_middle, BB_upper, BB_lower = bollinger_bands(close, BB_Length, BB_mult)
    BB_width = np.zeros_like(close)
    for i in range(n):
        if BB_middle[i] > 0 and not np.isnan(BB_upper[i]) and not np.isnan(BB_lower[i]):
            BB_width[i] = (BB_upper[i] - BB_lower[i]) / BB_middle[i]
        else:
            BB_width[i] = np.nan
    
    # Volume SMA
    vol_sma = sma(volume, sma_Length)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, atrPeriodSl)
    
    # Donchian Channel
    DCupper = rolling_max(high, DClength)
    DClower = rolling_min(low, DClength)
    
    # State tracking for position management
    position = 0  # 0 = flat, 1 = long, -1 = short
    avg_price = 0.0
    
    # Warmup period (use max of all indicator lengths)
    warmup = max(sEma_Length, ADX_len + ADX_smo, BB_Length, DClength, atrPeriodSl, slowLength)
    
    for i in range(n):
        # Skip warmup period
        if i < warmup:
            signals[i] = 0
            continue
        
        # EMA conditions
        EMA_longCond = fEMA[i] > sEMA[i] and (i > 0 and sEMA[i] > sEMA[i-1])
        EMA_shortCond = fEMA[i] < sEMA[i] and (i > 0 and sEMA[i] < sEMA[i-1])
        
        # ADX conditions
        ADX_longCond = DIPlus[i] > DIMinus[i] and ADX[i] > th
        ADX_shortCond = DIPlus[i] < DIMinus[i] and ADX[i] > th
        
        # SAR conditions
        SAR_longCond = SAR[i] < close[i]
        SAR_shortCond = SAR[i] > close[i]
        
        # MACD conditions (using standard MACD histogram)
        MACD_longCond = hist[i] > 0
        MACD_shortCond = hist[i] < 0
        
        # Volume condition
        VOL_longCond = volume[i] > vol_sma[i] * volume_f
        VOL_shortCond = VOL_longCond
        
        # Bollinger Bands conditions
        BB_long01 = (not ADX_shortCond) and (low[i] < BB_lower[i]) and EMA_longCond and BB_width[i] > (bbMinWidth01 / 100)
        BB_long02 = (not ADX_shortCond) and (low[i] < BB_lower[i]) and EMA_longCond and BB_width[i] > (bbMinWidth02 / 100)
        
        BB_short01 = (not ADX_longCond) and (high[i] > BB_upper[i]) and EMA_shortCond and BB_width[i] > (bbMinWidth01 / 100)
        BB_short02 = (not ADX_longCond) and (high[i] > BB_upper[i]) and EMA_shortCond and BB_width[i] > (bbMinWidth02 / 100)
        
        # Main entry conditions
        longCond = EMA_longCond and ADX_longCond and SAR_longCond and MACD_longCond and VOL_longCond
        shortCond = EMA_shortCond and ADX_shortCond and SAR_shortCond and MACD_shortCond and VOL_shortCond
        
        # Calculate ATR-based stop loss levels
        ATR_SL_Long = low[i] - atr[i] * multiplierPeriodSl
        ATR_SL_Short = high[i] + atr[i] * multiplierPeriodSl
        
        # Trailing stop logic for ATR
        if i > 0 and position == 1:
            longStopPrev = prev_atr_sl_long
            if open_price[i] > longStopPrev:
                ATR_SL_Long = max(ATR_SL_Long, longStopPrev)
        if i > 0 and position == -1:
            shortStopPrev = prev_atr_sl_short
            if open_price[i] < shortStopPrev:
                ATR_SL_Short = min(ATR_SL_Short, shortStopPrev)
        
        prev_atr_sl_long = ATR_SL_Long
        prev_atr_sl_short = ATR_SL_Short
        
        # Stop loss levels (use max/min of ATR and percentage)
        longPriceStop = max(ATR_SL_Long, (1 - (sl / 100)) * avg_price) if avg_price > 0 else ATR_SL_Long
        shortPriceStop = min(ATR_SL_Short, (1 + (sl / 100)) * avg_price) if avg_price > 0 else ATR_SL_Short
        
        # Take profit levels
        longPriceProfit = max(DCupper[i] if not np.isnan(DCupper[i]) else 0, (1 + (tp / 100)) * avg_price) if avg_price > 0 else DCupper[i]
        shortPriceProfit = min(DClower[i] if not np.isnan(DClower[i]) else 0, (1 - (tp / 100)) * avg_price) if avg_price > 0 else DClower[i]
        
        # Exit logic (check stops/profits using current bar high/low)
        if position == 1:
            if low[i] <= longPriceStop or high[i] >= longPriceProfit:
                position = 0
                signals[i] = 0
                avg_price = 0.0
                continue
        elif position == -1:
            if high[i] >= shortPriceStop or low[i] <= shortPriceProfit:
                position = 0
                signals[i] = 0
                avg_price = 0.0
                continue
        
        # Entry logic (next-bar execution - signal set today, fill at next bar open)
        if position == 0:
            if longCond or BB_long01:
                position = 1
                signals[i] = 1
                avg_price = close[i]
            elif shortCond or BB_short01:
                position = -1
                signals[i] = -1
                avg_price = close[i]
        elif position == 1:
            # Maintain long or pyramid on better price
            if (longCond or BB_long01) and close[i] < avg_price * (1 - (bbBetterPrice / 100)):
                signals[i] = 1
                avg_price = (avg_price + close[i]) / 2
            else:
                signals[i] = 1
        elif position == -1:
            # Maintain short or pyramid on better price
            if (shortCond or BB_short01) and close[i] > avg_price * (1 + (bbBetterPrice / 100)):
                signals[i] = -1
                avg_price = (avg_price + close[i]) / 2
            else:
                signals[i] = -1
    
    return signals


# Helper functions
def ema(data, length):
    """Calculate Exponential Moving Average"""
    result = np.zeros_like(data)
    if length <= 0:
        return result
    multiplier = 2.0 / (length + 1)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
    return result


def sma(data, length):
    """Calculate Simple Moving Average"""
    result = np.full_like(data, np.nan)
    if length <= 0:
        return result
    for i in range(len(data)):
        if i >= length - 1:
            result[i] = np.mean(data[i-length+1:i+1])
    return result


def stdev(data, length):
    """Calculate Standard Deviation (population)"""
    result = np.full_like(data, np.nan)
    if length <= 0:
        return result
    for i in range(len(data)):
        if i >= length - 1:
            result[i] = np.std(data[i-length+1:i+1], ddof=0)
    return result


def macd(data, fast, slow, signal):
    """Calculate MACD"""
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def dmi(close, high, low, length, smooth):
    """Calculate ADX/DMI"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    
    plus_sum = np.zeros(n)
    minus_sum = np.zeros(n)
    tr_sum = np.zeros(n)
    
    for i in range(n):
        if i < length:
            plus_sum[i] = np.sum(plus_dm[:i+1])
            minus_sum[i] = np.sum(minus_dm[:i+1])
            tr_sum[i] = np.sum(tr[:i+1])
        else:
            plus_sum[i] = plus_sum[i-1] - plus_sum[i-1]/length + plus_dm[i]
            minus_sum[i] = minus_sum[i-1] - minus_sum[i-1]/length + minus_dm[i]
            tr_sum[i] = tr_sum[i-1] - tr_sum[i-1]/length + tr[i]
        
        if tr_sum[i] > 0:
            plus_di[i] = 100 * plus_sum[i] / tr_sum[i]
            minus_di[i] = 100 * minus_sum[i] / tr_sum[i]
    
    dx = np.zeros(n)
    for i in range(n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    for i in range(n):
        if i < smooth:
            adx[i] = np.mean(dx[:i+1]) if i > 0 else dx[i]
        else:
            adx[i] = (adx[i-1] * (smooth - 1) + dx[i]) / smooth
    
    return plus_di, minus_di, adx


def parabolic_sar(high, low, start, increment, max_val):
    """Calculate Parabolic SAR"""
    n = len(high)
    sar = np.zeros(n)
    trend = 1
    ep = high[0]
    af = start
    sar[0] = low[0]
    
    for i in range(1, n):
        if trend == 1:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + increment, max_val)
            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = start
        else:
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + increment, max_val)
            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = start
    
    return sar


def bollinger_bands(data, length, mult):
    """Calculate Bollinger Bands"""
    middle = sma(data, length)
    std = stdev(data, length)
    upper = middle + mult * std
    lower = middle - mult * std
    return middle, upper, lower


def calculate_atr(high, low, close, length):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(n):
        if i < length:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
    
    return atr


def rolling_max(data, length):
    """Calculate rolling maximum"""
    result = np.full_like(data, np.nan)
    for i in range(len(data)):
        if i >= length - 1:
            result[i] = np.max(data[i-length+1:i+1])
    return result


def rolling_min(data, length):
    """Calculate rolling minimum"""
    result = np.full_like(data, np.nan)
    for i in range(len(data)):
        if i >= length - 1:
            result[i] = np.min(data[i-length+1:i+1])
    return result


def calc_zvwap(close, volume, length):
    """Calculate Z-score of VWAP"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if i >= length - 1:
            vw = np.sum(volume[i-length+1:i+1] * close[i-length+1:i+1])
            v = np.sum(volume[i-length+1:i+1])
            if v > 0:
                mean = vw / v
                variance = np.mean((close[i-length+1:i+1] - mean) ** 2)
                std = np.sqrt(variance)
                if std > 0:
                    result[i] = (close[i] - mean) / std
    
    return result
