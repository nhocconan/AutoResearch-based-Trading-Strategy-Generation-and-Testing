#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R momentum with 1d EMA34 trend filter and volume confirmation
# Williams %R measures overbought/oversold levels (-100 to 0)
# Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 (bullish regime)
# Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 (bearish regime)
# Volume confirmation ensures momentum validity
# Works in both bull and bear markets by following 1d trend while capturing 12h mean-reversion swings
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100.0,
        -50.0  # neutral when range is zero
    )
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
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
        prev_williams_r = williams_r[i-1]  # for crossover detection
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Williams %R crosses above -80 (oversold bounce) AND bullish regime
                if prev_williams_r <= -80.0 and curr_williams_r > -80.0 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 (overbought rejection) AND bearish regime
                elif prev_williams_r >= -20.0 and curr_williams_r < -20.0 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R crosses below -50 (momentum loss) OR regime changes to bearish
            if prev_williams_r > -50.0 and curr_williams_r <= -50.0:
                signals[i] = 0.0
                position = 0
            elif not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R crosses above -50 (momentum loss) OR regime changes to bullish
            if prev_williams_r < -50.0 and curr_williams_r >= -50.0:
                signals[i] = 0.0
                position = 0
            elif not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals