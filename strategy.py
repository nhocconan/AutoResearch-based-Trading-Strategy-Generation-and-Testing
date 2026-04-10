#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) AND 12h EMA(50) > EMA(200) (bullish trend) AND 12h volume > 1.5x 20-bar avg
# - Short when Williams %R(14) crosses below -20 (overbought) AND 12h EMA(50) < EMA(200) (bearish trend) AND 12h volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to balance return and drawdown
# - Williams %R identifies overextended moves in both bull and bear markets
# - 12h EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Williams %R(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 12h EMA trend filter: EMA(50) vs EMA(200)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_12h = ema_50_12h > ema_200_12h
    ema_bearish_12h = ema_50_12h < ema_200_12h
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    volume_20_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * volume_20_avg_12h)
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_bullish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bullish_12h)
    ema_bearish_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_bearish_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_bullish_12h_aligned[i]) or
            np.isnan(ema_bearish_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i])):
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
            # Long when Williams %R crosses above -80 (from below) AND 12h bullish trend AND volume spike
            williams_r_prev = williams_r_aligned[i-1] if i > 0 else -100
            williams_r_curr = williams_r_aligned[i]
            long_signal = (williams_r_prev <= -80 and williams_r_curr > -80) and \
                          ema_bullish_12h_aligned[i] and \
                          vol_spike_12h_aligned[i]
            
            # Short when Williams %R crosses below -20 (from above) AND 12h bearish trend AND volume spike
            short_signal = (williams_r_prev >= -20 and williams_r_curr < -20) and \
                           ema_bearish_12h_aligned[i] and \
                           vol_spike_12h_aligned[i]
            
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion (-50)
            # Exit when Williams %R returns to -50 (mean reversion)
            williams_r_prev = williams_r_aligned[i-1] if i > 0 else -100
            williams_r_curr = williams_r_aligned[i]
            
            exit_long = position == 1 and williams_r_prev < -50 and williams_r_curr >= -50
            exit_short = position == -1 and williams_r_prev > -50 and williams_r_curr <= -50
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals