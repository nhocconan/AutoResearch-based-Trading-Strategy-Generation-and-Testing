#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# Jaw (13-period, 8-bar offset), Teeth (8-period, 5-bar offset), Lips (5-period, 3-bar offset)
# In trending markets: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
# In ranging markets: lines intertwine
# We use 1d ADX > 25 to filter for trending markets only
# Breakout signals: Go long when Lips crosses above Teeth with volume confirmation
# Go short when Lips crosses below Teeth with volume confirmation
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown

name = "12h_WilliamsAlligator_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (using standard Wilder's smoothing)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_values = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            smma_values[period-1] = sma[period-1]
            for i in range(period, len(data)):
                smma_values[i] = (smma_values[i-1] * (period-1) + data[i]) / period
        return smma_values
    
    jaw = smma(high, 13)  # Using high for jaw as per original Alligator
    teeth = smma(low, 8)   # Using low for teeth
    lips = smma(close, 5)  # Using close for lips
    
    # Williams Alligator signals: Lips crossing Teeth
    # Bullish: Lips crosses above Teeth
    # Bearish: Lips crosses below Teeth
    lips_above_teeth = lips > teeth
    lips_below_teeth = lips < teeth
    
    # Cross detection
    lips_cross_above_teeth = np.zeros(n, dtype=bool)
    lips_cross_below_teeth = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(lips[i-1]) and not np.isnan(teeth[i-1]):
            if lips[i-1] <= teeth[i-1] and lips[i] > teeth[i]:
                lips_cross_above_teeth[i] = True
            elif lips[i-1] >= teeth[i-1] and lips[i] < teeth[i]:
                lips_cross_below_teeth[i] = True
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Moderate threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending market filter)
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips crosses above Teeth with volume confirmation and trending market
            if lips_cross_above_teeth[i] and trending_market and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips crosses below Teeth with volume confirmation and trending market
            elif lips_cross_below_teeth[i] and trending_market and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Lips crosses below Teeth (reversal) OR market loses trend
            if lips_cross_below_teeth[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips crosses above Teeth (reversal) OR market loses trend
            if lips_cross_above_teeth[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# Jaw (13-period, 8-bar offset), Teeth (8-period, 5-bar offset), Lips (5-period, 3-bar offset)
# In trending markets: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
# In ranging markets: lines intertwine
# We use 1d ADX > 25 to filter for trending markets only
# Breakout signals: Go long when Lips crosses above Teeth with volume confirmation
# Go short when Lips crosses below Teeth with volume confirmation
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown

name = "12h_WilliamsAlligator_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (using standard Wilder's smoothing)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        smma_values = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            smma_values[period-1] = sma[period-1]
            for i in range(period, len(data)):
                smma_values[i] = (smma_values[i-1] * (period-1) + data[i]) / period
        return smma_values
    
    jaw = smma(high, 13)  # Using high for jaw as per original Alligator
    teeth = smma(low, 8)   # Using low for teeth
    lips = smma(close, 5)  # Using close for lips
    
    # Williams Alligator signals: Lips crossing Teeth
    # Bullish: Lips crosses above Teeth
    # Bearish: Lips crosses below Teeth
    lips_above_teeth = lips > teeth
    lips_below_teeth = lips < teeth
    
    # Cross detection
    lips_cross_above_teeth = np.zeros(n, dtype=bool)
    lips_cross_below_teeth = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(lips[i]) and not np.isnan(teeth[i]) and not np.isnan(lips[i-1]) and not np.isnan(teeth[i-1]):
            if lips[i-1] <= teeth[i-1] and lips[i] > teeth[i]:
                lips_cross_above_teeth[i] = True
            elif lips[i-1] >= teeth[i-1] and lips[i] < teeth[i]:
                lips_cross_below_teeth[i] = True
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Moderate threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending market filter)
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips crosses above Teeth with volume confirmation and trending market
            if lips_cross_above_teeth[i] and trending_market and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips crosses below Teeth with volume confirmation and trending market
            elif lips_cross_below_teeth[i] and trending_market and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Lips crosses below Teeth (reversal) OR market loses trend
            if lips_cross_below_teeth[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Lips crosses above Teeth (reversal) OR market loses trend
            if lips_cross_above_teeth[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals