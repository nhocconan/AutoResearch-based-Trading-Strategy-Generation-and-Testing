#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout direction
# filtered by 1d EMA34 trend and confirmed by volume spike (>2x 20-period EMA).
# Works in bull/bear markets: buys breakouts above upper BB in uptrends,
# sells breakdowns below lower BB in downtrends. Avoids whipsaws in ranging markets
# by requiring both squeeze breakout and volume confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).

name = "6h_BollingerSqueeze_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    bb_width = (upper_bb - lower_bb) / sma_bb  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period EMA of width
    bb_width_series = pd.Series(bb_width)
    bb_width_ema = bb_width_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ema
    
    # Volume confirmation: 2.0x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Bollinger squeeze breakout above upper BB + volume confirmation + price above 1d EMA34 (uptrend)
            if (squeeze_condition[i] and close[i] > upper_bb[i] and volume_confirmed and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze breakout below lower BB + volume confirmation + price below 1d EMA34 (downtrend)
            elif (squeeze_condition[i] and close[i] < lower_bb[i] and volume_confirmed and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands (mean reversion) OR trend change
            if close[i] < sma_bb[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands (mean reversion) OR trend change
            if close[i] > sma_bb[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals