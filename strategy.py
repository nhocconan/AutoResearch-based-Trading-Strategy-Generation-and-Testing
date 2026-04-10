#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d EMA(50) > EMA(200) (bullish trend) AND 1d volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 1d EMA(50) < EMA(200) (bearish trend) AND 1d volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Williams %R identifies overextended moves likely to reverse
# - 1d EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: mean reversion in ranges, trend-following in strong moves

name = "12h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_1d = ema_50_1d > ema_200_1d
    ema_bearish_1d = ema_50_1d < ema_200_1d
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * volume_20_avg_1d)
    
    # Pre-compute 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Williams %R thresholds
    oversold = williams_r < -80
    overbought = williams_r > -20
    exit_signal = (williams_r >= -50) & (williams_r <= -50)  # Will be refined in loop
    
    # Align HTF indicators to 12h timeframe
    ema_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish_1d)
    ema_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    oversold_aligned = align_htf_to_ltf(prices, df_1d, oversold)
    overbought_aligned = align_htf_to_ltf(prices, df_1d, overbought)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1d_aligned[i]) or np.isnan(ema_bearish_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(oversold_aligned[i]) or
            np.isnan(overbought_aligned[i]) or np.isnan(williams_r_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when Williams %R oversold AND 1d bullish trend AND volume spike
            if (oversold_aligned[i] and 
                ema_bullish_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R overbought AND 1d bearish trend AND volume spike
            elif (overbought_aligned[i] and 
                  ema_bearish_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion
            # Exit when Williams %R returns to -50 (mean reversion midpoint)
            exit_long = position == 1 and williams_r_aligned[i] >= -50
            exit_short = position == -1 and williams_r_aligned[i] <= -50
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals