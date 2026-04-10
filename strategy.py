#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 (1d) AND 1w EMA(34) > EMA(89) (bullish trend) AND 1d volume > 2.0x 20-bar avg
# - Short when price breaks below Camarilla L4 (1d) AND 1w EMA(34) < EMA(89) (bearish trend) AND 1d volume > 2.0x 20-bar avg
# - Exit when price returns to Camarilla pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Camarilla H4/L4 represent stronger breakout levels than H3/L3, reducing false signals
# - 1w EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation avoids low-liquidity false signals
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_camarilla_breakout_volume_trend_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA trend filter: EMA(34) vs EMA(89)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, min_periods=89, adjust=False).mean().values
    ema_bullish_1w = ema_34_1w > ema_89_1w
    ema_bearish_1w = ema_34_1w < ema_89_1w
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = prices['volume'].values
    volume_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * volume_20_avg_1d)
    
    # Pre-compute 1d Camarilla pivot levels from prior day
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Camarilla calculations (using prior day's OHLC)
    rang = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = camarilla_pivot + (rang * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (rang * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (rang * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (rang * 1.1 / 2)
    
    # Align HTF indicators to 1d timeframe
    ema_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish_1w)
    ema_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish_1w)
    
    # Session filter: 08-20 UTC (avoid low liquidity Asian session)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_1w_aligned[i]) or np.isnan(ema_bearish_1w_aligned[i]) or
            np.isnan(vol_spike_1d[i]) or np.isnan(camarilla_pivot[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Apply session filter
        if not in_session[i]:
            # Outside session: flatten position
            position = 0
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H4 AND 1w bullish trend AND volume spike
            if (prices['close'].iloc[i] > camarilla_h4[i] and 
                ema_bullish_1w_aligned[i] and 
                vol_spike_1d[i]):
                position = 1
                signals[i] = 0.30
            # Short when price breaks below L4 AND 1w bearish trend AND volume spike
            elif (prices['close'].iloc[i] < camarilla_l4[i] and 
                  ema_bearish_1w_aligned[i] and 
                  vol_spike_1d[i]):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to Camarilla pivot point
            exit_long = position == 1 and prices['close'].iloc[i] <= camarilla_pivot[i]
            exit_short = position == -1 and prices['close'].iloc[i] >= camarilla_pivot[i]
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.30
                else:
                    signals[i] = -0.30
    
    return signals