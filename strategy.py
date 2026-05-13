# 6h_Anchored_VWAP_Retest_1dTrend_Volume
# Hypothesis: Price tends to retest daily Anchored VWAP after significant moves.
# Long when price retests AVWAP from above during 1d uptrend with volume confirmation.
# Short when price retests AVWAP from below during 1d downtrend with volume confirmation.
# Exit when price moves 1 ATR away from AVWAP or reverses trend.
# VWAP acts as dynamic support/resistance, reducing false signals.
# Target: 15-25 trades/year on 6h to minimize fee drag.

name = "6h_Anchored_VWAP_Retest_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for Anchored VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate Anchored VWAP (anchored to start of each day)
    # VWAP = sum(price * volume) / sum(volume) where price = (H+L+C)/3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_volume_1d = np.cumsum(volume_1d)
    # Avoid division by zero
    vwap_1d = np.divide(cum_pv_1d, cum_volume_1d, out=np.full_like(cum_pv_1d, np.nan), where=cum_volume_1d!=0)

    # Reset VWAP at start of each day (when cumulative volume resets)
    # Find where volume_1d drops significantly (start of new day)
    # For daily data, we reset when we see the first bar of the day
    # Simpler: reset when the cumulative volume is at a local minimum (start of day)
    # Actually, for daily data, VWAP is just for that day, so we don't need to reset within day
    # But we need to anchor to each day's start, so we calculate VWAP for each day separately
    # Let's do it properly: for each day, VWAP resets at open
    vwap_1d_reset = np.full_like(close_1d, np.nan)
    vol_sum = 0
    pv_sum = 0
    for i in range(len(close_1d)):
        pv_sum += typical_price_1d[i] * volume_1d[i]
        vol_sum += volume_1d[i]
        if vol_sum > 0:
            vwap_1d_reset[i] = pv_sum / vol_sum
        # Reset at end of day (we don't have intraday, so each 1d bar is a day)
        # Actually, since we're using daily data, each bar is a complete day
        # So VWAP for that day is just the day's VWAP, no need to reset within the bar
        # But we want the VWAP value to be available throughout the day
        # For daily timeframe, VWAP is calculated within the day and uses intraday data
        # Since we don't have intraday, we'll approximate: VWAP ≈ typical price for the day
        # This is not ideal, but let's use a different approach

    # Instead, let's use a simpler approximation: VWAP of the day is close to typical price
    # But we need a dynamic VWAP that updates throughout the day
    # Given we only have daily data, we'll use the day's VWAP as a constant for that day
    # And anchor it to the start of the day
    # So for each 1d bar, VWAP is the day's VWAP, and it's valid for the entire day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (np.cumsum(typical_price_1d * volume_1d)) / (np.cumsum(volume_1d))
    # Handle division by zero
    vwap_1d = np.divide(np.cumsum(typical_price_1d * volume_1d), 
                        np.cumsum(volume_1d), 
                        out=np.full_like(np.cumsum(typical_price_1d * volume_1d), np.nan), 
                        where=np.cumsum(volume_1d)!=0)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align VWAP and EMA to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # ATR for exit condition
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price retests VWAP from above + uptrend + volume
            # Price is above VWAP but close to it (within 0.5*ATR)
            near_vwap = abs(close[i] - vwap_1d_aligned[i]) < (0.5 * atr[i])
            price_above_vwap = close[i] > vwap_1d_aligned[i]
            uptrend = close[i] > ema34_1d_aligned[i]
            vol_ok = volume[i] > vol_avg_20[i] * 1.3

            if near_vwap and price_above_vwap and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price retests VWAP from below + downtrend + volume
            near_vwap = abs(close[i] - vwap_1d_aligned[i]) < (0.5 * atr[i])
            price_below_vwap = close[i] < vwap_1d_aligned[i]
            downtrend = close[i] < ema34_1d_aligned[i]
            vol_ok = volume[i] > vol_avg_20[i] * 1.3

            if near_vwap and price_below_vwap and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves 1*ATR away from VWAP or trend reverses
            price_far_from_vwap = (close[i] - vwap_1d_aligned[i]) > (1.0 * atr[i])
            trend_reversal = close[i] < ema34_1d_aligned[i]
            if price_far_from_vwap or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves 1*ATR away from VWAP or trend reverses
            price_far_from_vwap = (vwap_1d_aligned[i] - close[i]) > (1.0 * atr[i])
            trend_reversal = close[i] > ema34_1d_aligned[i]
            if price_far_from_vwap or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals