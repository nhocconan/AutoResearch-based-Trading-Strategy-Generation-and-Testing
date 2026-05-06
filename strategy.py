#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Volume Weighted Average Price (VWAP) with
# mean reversion at VWAP bands and trend filter. VWAP acts as dynamic support/resistance.
# Mean reversion: long when price touches VWAP lower band with rejection, short at upper band.
# Trend filter: only take trades in direction of 1-day EMA50 to avoid counter-trend in strong trends.
# Volume confirmation: require volume > 1.5x 20-period average to confirm interest.
# Target: 20-50 trades/year with 0.30 position sizing to minimize fee drag.

name = "4h_VWAP_MeanReversion_EMA50_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for each 1d bar
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Reset VWAP calculation at each new day (when cumulative volume resets)
    # Handle case where volume is zero at start of day
    vol_reset = df_1d['volume'] == 0
    vwap = vwap.where(~vol_reset, typical_price)
    
    # Calculate VWAP bands (1 ATR width)
    atr_1d = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).mean()
    vwap_upper = vwap + atr_1d
    vwap_lower = vwap - atr_1d
    
    # Align VWAP and bands to 4h timeframe
    vwap_4h = align_htf_to_ltf(prices, df_1d, vwap.values)
    vwap_upper_4h = align_htf_to_ltf(prices, df_1d, vwap_upper.values)
    vwap_lower_4h = align_htf_to_ltf(prices, df_1d, vwap_lower.values)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vwap_4h[i]) or np.isnan(vwap_upper_4h[i]) or np.isnan(vwap_lower_4h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion long at VWAP lower band: price touches lower band and closes above it
            if close[i] <= vwap_lower_4h[i] * 1.002 and close[i] > vwap_lower_4h[i] and volume_filter[i]:
                # Only take long if above daily EMA50 (bullish bias)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.30
                    position = 1
            # Mean reversion short at VWAP upper band: price touches upper band and closes below it
            elif close[i] >= vwap_upper_4h[i] * 0.998 and close[i] < vwap_upper_4h[i] and volume_filter[i]:
                # Only take short if below daily EMA50 (bearish bias)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Exit long: price reaches VWAP (mean reversion target) or breaks above upper band (stop)
            if close[i] >= vwap_4h[i] * 0.999:  # Take profit at VWAP
                signals[i] = 0.0
                position = 0
            elif close[i] > vwap_upper_4h[i]:  # Stop loss if breaks above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price reaches VWAP (mean reversion target) or breaks below lower band (stop)
            if close[i] <= vwap_4h[i] * 1.001:  # Take profit at VWAP
                signals[i] = 0.0
                position = 0
            elif close[i] < vwap_lower_4h[i]:  # Stop loss if breaks below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals