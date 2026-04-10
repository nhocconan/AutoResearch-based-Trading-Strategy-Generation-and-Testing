#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and ADX trend filter
# - Primary: 4h Williams %R(14) < -80 for long, > -20 for short (oversold/overbought)
# - Volume filter: 1d volume > 2.0x 20-period volume MA for institutional confirmation
# - Trend filter: ADX(14) > 20 to avoid extremely choppy markets (adaptive threshold)
# - Exit: Williams %R returns to -50 level (mean reversion)
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Works in bull/bear: Williams %R identifies extremes in any market, volume confirms
#   participation, ADX filter avoids whipsaws in low-volatility regimes

name = "4h_1d_williamsr_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R for 4h timeframe (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d volume confirmation: volume > 2.0x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate ADX(14) for trend filter
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period MA (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 20 (avoid extremely choppy markets)
        trend_filter = adx[i] > 20
        
        if position == 0:  # Flat - look for new Williams %R extremes
            # Long entry: Williams %R < -80 (oversold) + vol confirmation + trend filter
            if williams_r[i] < -80 and vol_confirm and trend_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + vol confirmation + trend filter
            elif williams_r[i] > -20 and vol_confirm and trend_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion
            # Exit: Williams %R returns to -50 level (mean reversion)
            if position == 1:  # Long position
                if williams_r[i] >= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals