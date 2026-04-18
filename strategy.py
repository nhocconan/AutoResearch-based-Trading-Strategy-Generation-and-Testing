#!/usr/bin/env python3
"""
1d Bollinger Band Width + RSI Mean Reversion with 1w Trend Filter
Strategy: In low volatility (BBW < 20th percentile), look for mean reversion:
          Long when RSI < 30 and price > 1w EMA200
          Short when RSI > 70 and price < 1w EMA200
          Uses weekly EMA200 as trend filter to align with higher timeframe direction.
          Designed for low trade frequency with clear mean reversion edge in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    weekly_close = df_1w['close'].values
    ema_200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mult = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = close_series.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_middle + (bb_std * bb_mult)
    bb_lower = bb_middle - (bb_std * bb_mult)
    bb_width = bb_upper - bb_lower
    bb_width_values = bb_width.values
    
    # Calculate 50-period percentile rank of BBW for volatility regime
    bb_width_series = pd.Series(bb_width_values)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # RSI (14)
    rsi_period = 14
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill NaN with 50 for start
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need enough history for weekly EMA200 and other indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bbw_percentile = bb_width_percentile[i]
        rsi_val = rsi_values[i]
        ema_200 = ema_200_1w_aligned[i]
        
        # Low volatility regime: BBW < 20th percentile
        if bbw_percentile < 20:
            if position == 0:
                # Long: RSI oversold and price above weekly EMA200
                if rsi_val < 30 and price > ema_200:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI overbought and price below weekly EMA200
                elif rsi_val > 70 and price < ema_200:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Long position management
                signals[i] = 0.25
                # Exit: RSI returns to neutral (50) or volatility increases
                if rsi_val >= 50 or bbw_percentile >= 30:
                    signals[i] = 0.0
                    position = 0
            
            elif position == -1:
                # Short position management
                signals[i] = -0.25
                # Exit: RSI returns to neutral (50) or volatility increases
                if rsi_val <= 50 or bbw_percentile >= 30:
                    signals[i] = 0.0
                    position = 0
        else:
            # High volatility: stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_BBW_RSI_MeanReversion_1wEMA200"
timeframe = "1d"
leverage = 1.0