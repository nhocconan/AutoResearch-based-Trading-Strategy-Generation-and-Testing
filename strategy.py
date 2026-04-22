#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB after low volatility squeeze, with 1d uptrend and volume spike
# Short when price breaks below lower BB after low volatility squeeze, with 1d downtrend and volume spike
# Bollinger Band width percentile identifies low volatility periods (squeeze) that precede explosive moves
# Designed for 4h timeframe to target 20-30 trades/year per symbol.
# Works in both bull and bear markets by using 1d trend filter to avoid counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20, 2) on 4h data
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band width percentile (identify squeeze)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    squeeze = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB after squeeze + 1d uptrend + volume spike
            if (close[i] > upper_band[i] and squeeze[i-1] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB after squeeze + 1d downtrend + volume spike
            elif (close[i] < lower_band[i] and squeeze[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to middle Bollinger Band or trend reversal
            if position == 1:
                # Exit on price below middle BB or trend reversal
                if (close[i] < sma[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above middle BB or trend reversal
                if (close[i] > sma[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_BollingerSqueeze_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0