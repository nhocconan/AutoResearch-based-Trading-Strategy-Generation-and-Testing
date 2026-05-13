#!/usr/bin/env python3
# 6h_VWAP_Deviation_MeanReversion_1dTrend
# Hypothesis: Mean reversion to VWAP on 6h timeframe, filtered by 1d trend direction and volatility regime.
# In bull/bear markets, price tends to revert to VWAP during consolidation phases.
# Uses 1d EMA50 for trend filter (only long in uptrend, short in downtrend).
# Uses ATR-based volatility filter to avoid ranging markets.
# Designed to capture mean reversion moves with tight entries to minimize fee drag.
# Target: 15-30 trades/year per symbol.

name = "6h_VWAP_Deviation_MeanReversion_1dTrend"
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

    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate VWAP for 6h: cumulative (price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    # Reset VWAP at session start (assuming 6h bars align with sessions)
    # For simplicity, we'll use a rolling window approximation
    # But to be more accurate, we reset daily - however 6h bars may cross days
    # Instead, use typical price deviation from VWAP-like measure
    # Use 20-period VWAP approximation
    vwap_20 = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values

    # Deviation from VWAP as percentage
    deviation = (close - vwap_20) / vwap_20 * 100.0

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volatility filter: ATR(14) > 20-period average ATR (avoid low volatility)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr14 > (0.8 * atr_ma)  # Only trade when volatility is above 80% of average

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(deviation[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price deviates below VWAP (oversold) in uptrend with sufficient volatility
            if (deviation[i] < -1.5 and 
                close[i] > ema50_1d_aligned[i] and 
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price deviates above VWAP (overbought) in downtrend with sufficient volatility
            elif (deviation[i] > 1.5 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend turns down
            if deviation[i] > -0.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend turns up
            if deviation[i] < 0.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals