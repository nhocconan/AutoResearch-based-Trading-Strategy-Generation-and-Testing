# 1d_KAMA_RSI_ChopFilter_v1
# Hypothesis: 1d KAMA identifies adaptive trend direction, RSI filters for extreme mean-reversion opportunities,
# and Chop filter (via ADX) avoids whipsaws in strong trends. Works in bull/bear by capturing reversals
# in ranging markets while avoiding trend-following losses. Low trade frequency via strict confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA trend, RSI, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA (Adaptive Moving Average) calculation
    def calculate_kama(close_series, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close_series, n=er_len))
        volatility = np.sum(np.abs(np.diff(close_series)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1))**2
        kama = np.full_like(close_series, np.nan, dtype=float)
        kama[er_len] = close_series[er_len]
        for i in range(er_len+1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(df_1d['close'])
    kama_slope = np.diff(kama, prepend=kama[0])
    kama_trend = kama_slope > 0  # 1 for up, 0 for down
    
    # RSI(14)
    def calculate_rsi(close_series, period=14):
        delta = np.diff(close_series, prepend=close_series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close_series, np.nan, dtype=float)
        avg_loss = np.full_like(close_series, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(close_series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(df_1d['close'])
    
    # ADX(14) for Chop filter (ADX < 20 = ranging/chop)
    def calculate_adx(high_series, low_series, close_series, period=14):
        plus_dm = np.where((high_series[1:] - high_series[:-1]) > (low_series[:-1] - low_series[1:]), 
                           np.maximum(high_series[1:] - high_series[:-1], 0), 0)
        minus_dm = np.where((low_series[:-1] - low_series[1:]) > (high_series[1:] - high_series[:-1]), 
                            np.maximum(low_series[:-1] - low_series[1:], 0), 0)
        tr = np.maximum(high_series[1:] - low_series[1:], 
                        np.maximum(np.abs(high_series[1:] - close_series[:-1]), 
                                   np.abs(low_series[1:] - close_series[:-1])))
        tr = np.concatenate([[np.max([high_series[0] - low_series[0], 
                                       np.abs(high_series[0] - close_series[0]),
                                       np.abs(low_series[0] - close_series[0])])], tr])
        plus_di = 100 * (np.convolve(plus_dm, np.ones(period)/period, mode='same') / 
                         np.convolve(tr, np.ones(period)/period, mode='same'))
        minus_di = 100 * (np.convolve(minus_dm, np.ones(period)/period, mode='same') / 
                          np.convolve(tr, np.ones(period)/period, mode='same'))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = np.convolve(dx, np.ones(period)/period, mode='same')
        # Pad to match original length
        adx = np.concatenate([np.full(period-1, np.nan), adx[:len(close_series)-period+1]])
        return adx
    
    adx = calculate_adx(df_1d['high'], df_1d['low'], df_1d['close'])
    chop_filter = adx < 20  # Ranging market condition
    
    # Align all indicators to lower timeframe (1d -> 1d is identity, but we keep for consistency)
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA uptrend + RSI oversold (<30) + chop/ranging market
            if kama_trend_aligned[i] == 1 and rsi_aligned[i] < 30 and chop_filter_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI overbought (>70) + chop/ranging market
            elif kama_trend_aligned[i] == 0 and rsi_aligned[i] > 70 and chop_filter_aligned[i] == 1:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend changes
            if position == 1:
                if rsi_aligned[i] > 40 or kama_trend_aligned[i] == 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_aligned[i] < 60 or kama_trend_aligned[i] == 1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0