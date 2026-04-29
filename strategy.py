#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (> -20 = overbought, < -80 = oversold)
# Long when Williams %R < -80 (oversold) AND price > 1w EMA34 (bullish higher timeframe trend)
# Short when Williams %R > -20 (overbought) AND price < 1w EMA34 (bearish higher timeframe trend)
# Volume confirmation ensures mean reversion has momentum behind it
# Works in both bull and bear markets by aligning with 1w trend while capturing 1d mean reversion extremes
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag on daily timeframe

name = "1d_WilliamsR_1wEMA34_Trend_VolumeConfirmation_v1"
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
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe (completed 1w bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100.0,
        -50.0  # neutral when no range
    )
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20, 14)  # warmup for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema34 = ema34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1w EMA34, bearish if price < 1w EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: oversold Williams %R AND bullish regime
                if curr_williams_r < -80 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: overbought Williams %R AND bearish regime
                elif curr_williams_r > -20 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R rises above -50 (mean reversion complete) OR regime changes to bearish
            if curr_williams_r > -50 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R falls below -50 (mean reversion complete) OR regime changes to bullish
            if curr_williams_r < -50 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals