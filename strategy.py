#!/usr/bin/env python3
# 1h_RSI_Divergence_4hTrend_Volume
# Hypothesis: Use 1h RSI divergence for entry timing with 4h EMA trend filter and volume confirmation.
# Long: Bullish RSI divergence (price lower low, RSI higher low) + price above 4h EMA20 + volume spike.
# Short: Bearish RSI divergence (price higher high, RSI lower high) + price below 4h EMA20 + volume spike.
# Exit: Opposite RSI divergence or RSI crosses 50.
# Target: 20-40 trades/year on 1h to minimize fee drift while capturing momentum reversals.
# Works in both bull/bear: divergence signals reversals, trend filter avoids counter-trend trades.

name = "1h_RSI_Divergence_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # 1h RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # Calculate RSI peaks and troughs for divergence detection
    # For bullish divergence: look for lower low in price with higher low in RSI
    # For bearish divergence: look for higher high in price with lower high in RSI
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    # Lookback period for divergence detection
    lookback = 10

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(rsi[i-1]) or np.isnan(avg_gain[i]) or np.isnan(avg_loss[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if i >= lookback:
                # Find recent price low and RSI low
                price_window = low[i-lookback:i+1]
                rsi_window = rsi[i-lookback:i+1]
                if len(price_window) >= 2 and len(rsi_window) >= 2:
                    price_min_idx = np.argmin(price_window)
                    rsi_min_idx = np.argmin(rsi_window)
                    # Price made a lower low but RSI made a higher low
                    if (price_min_idx == lookback and rsi_min_idx < lookback and
                        price_window[0] < price_window[-1] and rsi_window[0] > rsi_window[-1]):
                        bullish_div = True

            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if i >= lookback:
                # Find recent price high and RSI high
                price_window = high[i-lookback:i+1]
                rsi_window = rsi[i-lookback:i+1]
                if len(price_window) >= 2 and len(rsi_window) >= 2:
                    price_max_idx = np.argmax(price_window)
                    rsi_max_idx = np.argmax(rsi_window)
                    # Price made a higher high but RSI made a lower high
                    if (price_max_idx == lookback and rsi_max_idx < lookback and
                        price_window[0] > price_window[-1] and rsi_window[0] < rsi_window[-1]):
                        bearish_div = True

            # Volume confirmation: volume > 1.5x 20-period average
            vol_avg_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else volume[i]
            volume_spike = volume[i] > vol_avg_20 * 1.5

            # LONG: Bullish divergence + price above 4h EMA20 + volume spike
            if bullish_div and close[i] > ema20_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # SHORT: Bearish divergence + price below 4h EMA20 + volume spike
            elif bearish_div and close[i] < ema20_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or RSI crosses below 50
            bearish_exit = False
            if i >= lookback:
                price_window = high[i-lookback:i+1]
                rsi_window = rsi[i-lookback:i+1]
                if len(price_window) >= 2 and len(rsi_window) >= 2:
                    price_max_idx = np.argmax(price_window)
                    rsi_max_idx = np.argmax(rsi_window)
                    if (price_max_idx == lookback and rsi_max_idx < lookback and
                        price_window[0] > price_window[-1] and rsi_window[0] < rsi_window[-1]):
                        bearish_exit = True
            rsi_cross_down = rsi[i] < 50 and rsi[i-1] >= 50
            if bearish_exit or rsi_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Bullish divergence or RSI crosses above 50
            bullish_exit = False
            if i >= lookback:
                price_window = low[i-lookback:i+1]
                rsi_window = rsi[i-lookback:i+1]
                if len(price_window) >= 2 and len(rsi_window) >= 2:
                    price_min_idx = np.argmin(price_window)
                    rsi_min_idx = np.argmin(rsi_window)
                    if (price_min_idx == lookback and rsi_min_idx < lookback and
                        price_window[0] < price_window[-1] and rsi_window[0] > rsi_window[-1]):
                        bullish_exit = True
            rsi_cross_up = rsi[i] > 50 and rsi[i-1] <= 50
            if bullish_exit or rsi_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals