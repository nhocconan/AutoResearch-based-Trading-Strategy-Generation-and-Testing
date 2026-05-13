# 4H_CAMARILLA_R3_S3_BREAKOUT_TREND_MOMENTUM
# Hypothesis: Breakouts at Camarilla R3/S3 levels on 4h chart with momentum confirmation (RSI>55 for long, RSI<45 for short) and volume filter.
# Uses 1d trend filter (EMA50) to ensure trades align with higher timeframe trend.
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe.
# Works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend).

name = "4H_CAMARILLA_R3_S3_BREAKOUT_TREND_MOMENTUM"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar's range)
    # R3 = close + 1.1*(high-low)/4
    # S3 = close - 1.1*(high-low)/4
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Avoid NaN from shift
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r3 = np.full_like(prev_close, np.nan)
    camarilla_s3 = np.full_like(prev_close, np.nan)
    
    camarilla_r3[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 4
    camarilla_s3[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Momentum confirmation: RSI(14)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral RSI when undefined
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        if position == 0:
            # LONG: Price breaks above R3 with momentum (RSI>55) and volume in uptrend (price > EMA50)
            if camarilla_r3_aligned[i] > 0 and not np.isnan(camarilla_r3_aligned[i]) and \
               high[i] > camarilla_r3_aligned[i] and rsi_values[i] > 55 and volume_confirmed[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with momentum (RSI<45) and volume in downtrend (price < EMA50)
            elif camarilla_s3_aligned[i] > 0 and not np.isnan(camarilla_s3_aligned[i]) and \
                 low[i] < camarilla_s3_aligned[i] and rsi_values[i] < 45 and volume_confirmed[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R3 or momentum fades (RSI<50) or trend weakens
            if camarilla_r3_aligned[i] > 0 and not np.isnan(camarilla_r3_aligned[i]) and \
               (low[i] < camarilla_r3_aligned[i] or rsi_values[i] < 50 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S3 or momentum fades (RSI>50) or trend weakens
            if camarilla_s3_aligned[i] > 0 and not np.isnan(camarilla_s3_aligned[i]) and \
               (high[i] > camarilla_s3_aligned[i] or rsi_values[i] > 50 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals