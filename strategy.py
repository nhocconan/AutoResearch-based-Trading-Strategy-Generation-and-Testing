#!/usr/bin/env python3
# 1d_1w_RSI50_Trend_Follow
# Hypothesis: Follows weekly trend using RSI(14) on weekly timeframe for direction, enters on daily when RSI(14) crosses 50 in trend direction with volume confirmation. Exits when weekly RSI reverses or daily RSI crosses opposite level. Weekly trend filter avoids whipsaws in ranging markets, while daily entries capture momentum with lower frequency. Works in bull/bear by following weekly momentum. Targets 15-25 trades/year to minimize fee drag.

name = "1d_1w_RSI50_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI(14) for trend direction ---
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_1w)  # handle no loss case
    rsi_50_1w = rsi_1w - 50  # center around 0
    rsi_50_1w_slope = rsi_50_1w - np.roll(rsi_50_1w, 1)
    rsi_50_1w_slope[0] = 0
    rsi_50_1w_slope = pd.Series(rsi_50_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    rsi_50_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_50_1w)
    rsi_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, rsi_50_1w_slope)
    
    # --- Daily RSI(14) for entry/exit timing ---
    delta_d = np.diff(close, prepend=close[0])
    gain_d = np.where(delta_d > 0, delta_d, 0)
    loss_d = np.where(delta_d < 0, -delta_d, 0)
    avg_gain_d = pd.Series(gain_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_d = pd.Series(loss_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_d = avg_gain_d / (avg_loss_d + 1e-10)
    rsi_d = 100 - (100 / (1 + rs_d))
    rsi_d = np.where(avg_loss_d == 0, 100, rsi_d)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI calculation (14) + smoothing (3)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_50_1w_aligned[i]) or
            np.isnan(rsi_50_1w_slope_aligned[i]) or
            np.isnan(rsi_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend direction from centered RSI slope
        weekly_uptrend = rsi_50_1w_slope_aligned[i] > 0
        weekly_downtrend = rsi_50_1w_slope_aligned[i] < 0
        
        if position == 0:
            if weekly_uptrend and vol_surge[i]:
                # Long: weekly uptrend + volume surge + daily RSI crosses above 50
                if rsi_d[i] > 50 and rsi_d[i-1] <= 50:
                    signals[i] = 0.25
                    position = 1
            elif weekly_downtrend and vol_surge[i]:
                # Short: weekly downtrend + volume surge + daily RSI crosses below 50
                if rsi_d[i] < 50 and rsi_d[i-1] >= 50:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: weekly trend turns down OR daily RSI crosses below 50
                if weekly_downtrend or (rsi_d[i] < 50 and rsi_d[i-1] >= 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR daily RSI crosses above 50
                if weekly_uptrend or (rsi_d[i] > 50 and rsi_d[i-1] <= 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals