#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands squeeze + mean reversion
# - Enter long when price touches lower BB after squeeze (BBW < 20th percentile) and RSI < 30
# - Enter short when price touches upper BB after squeeze and RSI > 70
# - Uses weekly Bollinger Bands calculated from prior week's close
# - Designed to work in ranging markets (mean reversion at BB extremes) and low volatility periods
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_BollingerSqueeze_MeanReversion_WeeklyBB"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    weekly_close = df_1w['close'].values
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align weekly Bollinger Bands to daily timeframe
    bb_upper_daily = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_daily = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_width_percentile_daily = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    
    # Daily RSI for entry confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bb_upper_daily[i]) or np.isnan(bb_lower_daily[i]) or 
            np.isnan(bb_width_percentile_daily[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger squeeze condition: BB width below 20th percentile
        squeeze_condition = bb_width_percentile_daily[i] < 20
        
        if position == 0 and squeeze_condition:
            # Long entry: price touches lower BB and RSI oversold
            if close[i] <= bb_lower_daily[i] * 1.001 and close[i] >= bb_lower_daily[i] and rsi[i] < 30:
                if volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price touches upper BB and RSI overbought
            elif close[i] >= bb_upper_daily[i] * 0.999 and close[i] <= bb_upper_daily[i] and rsi[i] > 70:
                if volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches middle BB or RSI overbought
            if close[i] >= bb_middle[i] * 0.999 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches middle BB or RSI oversold
            if close[i] <= bb_middle[i] * 1.001 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals