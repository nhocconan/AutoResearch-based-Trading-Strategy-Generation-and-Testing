#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI + 1-week Stochastic RSI with Volume Confirmation
# - RSI(14) on 1d for momentum (oversold <30, overbought >70)
# - Stochastic RSI(14) on 1w to confirm momentum direction
# - Volume spike (>1.5x 20-period average) to filter false signals
# - Only trade in direction of weekly trend: long when weekly close > weekly open
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)
# - Works in bull/bear: RSI captures reversals, weekly trend filters counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Stochastic RSI and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Stochastic RSI on weekly data
    # RSI(14) on weekly close
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.rolling(window=14, min_periods=14).mean()
    avg_loss_1w = loss_1w.rolling(window=14, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = rsi_1w.fillna(50).values
    
    # Stochastic RSI: (RSI - min RSI) / (max RSI - min RSI) over 14 periods
    min_rsi_1w = pd.Series(rsi_1w).rolling(window=14, min_periods=14).min().values
    max_rsi_1w = pd.Series(rsi_1w).rolling(window=14, min_periods=14).max().values
    stoch_rsi_1w = (rsi_1w - min_rsi_1w) / (max_rsi_1w - min_rsi_1w)
    stoch_rsi_1w = np.where(max_rsi_1w - min_rsi_1w == 0, 0.5, stoch_rsi_1w)
    
    # Align Stochastic RSI to 1d timeframe
    stoch_rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, stoch_rsi_1w)
    
    # Weekly trend filter: bullish when weekly close > weekly open
    weekly_bullish = close_1w > df_1w['open'].values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Load daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate RSI(14) on daily data
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.rolling(window=14, min_periods=14).mean()
    avg_loss_1d = loss_1d.rolling(window=14, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.fillna(50).values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in indicators
        if np.isnan(rsi_1d[i]) or np.isnan(stoch_rsi_1w_aligned[i]) or np.isnan(weekly_bullish_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        rsi_oversold = rsi_1d[i] < 30
        rsi_overbought = rsi_1d[i] > 70
        stoch_bullish = stoch_rsi_1w_aligned[i] > 0.5
        stoch_bearish = stoch_rsi_1w_aligned[i] < 0.5
        vol_spike = volume_spike[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        
        if position == 0:
            # Long entry: RSI oversold + stoch bullish + volume spike + weekly bullish
            if rsi_oversold and stoch_bullish and vol_spike and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought + stoch bearish + volume spike + weekly bearish
            elif rsi_overbought and stoch_bearish and vol_spike and not weekly_bull:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or stoch bearish
            if rsi_overbought or stoch_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or stoch bullish
            if rsi_oversold or stoch_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_StochasticRSI_VolumeFilter"
timeframe = "1d"
leverage = 1.0