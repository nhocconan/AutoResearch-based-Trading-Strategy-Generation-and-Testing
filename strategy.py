# 4h_RSI_Trend_Reversal_1dADX_Filter
# Hypothesis: RSI extremes combined with trend filter on 1d ADX and volume confirmation.
# In bull markets, buy RSI<30 with 1d ADX>25 (trending) and volume spike.
# In bear markets, sell RSI>70 with 1d ADX>25 and volume spike.
# RSI mean reversion works in ranging markets, while ADX filter avoids chop.
# Volume spike confirms institutional participation at turning points.
# Designed for low turnover: only trade at extreme RSI with trend and volume confirmation.

name = "4h_RSI_Trend_Reversal_1dADX_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    # 1d ADX(14) for trend strength
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])

    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dpi_plus = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dpi_minus = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dpi_plus / atr
    di_minus = 100 * dpi_minus / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values

    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume spike: current > 2.0x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 30 (oversold) + ADX > 25 (trending) + volume spike
            if (rsi[i] < 30 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + ADX > 25 (trending) + volume spike
            elif (rsi[i] > 70 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete) or trend weakens
            if (rsi[i] > 50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 or trend weakens
            if (rsi[i] < 50 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals