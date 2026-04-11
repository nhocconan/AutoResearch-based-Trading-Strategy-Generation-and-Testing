#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike
# - Uses 1d timeframe with 1w HTF filter for major trend alignment
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
# - Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA(13)
# - Volume spike (>2x 20-period average) confirms institutional participation
# - Long: Alligator bullish (Lips>Teeth>Jaw) + Bull Power>0 + Volume spike
# - Short: Alligator bearish (Lips<Teeth<Jaw) + Bear Power<0 + Volume spike
# - Exit: Alligator reverses (Lips crosses Jaw) or volume drops below average
# - Discrete position sizing: ±0.30 to balance return and drawdown
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# - Works in bull markets via trend following and in bear markets via short signals

name = "1d_1w_alligator_elder_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for major trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for major trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Williams Alligator
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    # Teeth (Red): 8-period SMMA, shifted 5 bars  
    # Lips (Green): 5-period SMMA, shifted 3 bars
    # SMMA = Smoothed Moving Average (Wilder's smoothing)
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Pre-compute 1d Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Pre-compute 1d volume confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Major trend filter: price above/below 1w EMA(50)
        uptrend_1w = close_price > ema_50_1w_aligned[i]
        downtrend_1w = close_price < ema_50_1w_aligned[i]
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator bullish + Bull Power positive + Volume spike + 1w uptrend
        if alligator_bullish and bull_strong and vol_confirm and uptrend_1w:
            enter_long = True
        
        # Short: Alligator bearish + Bear Power negative + Volume spike + 1w downtrend
        if alligator_bearish and bear_strong and vol_confirm and downtrend_1w:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Alligator turns bearish or volume drops
            exit_long = (not alligator_bullish) or (not vol_confirm)
        elif position == -1:
            # Exit short if Alligator turns bullish or volume drops
            exit_short = (not alligator_bearish) or (not vol_confirm)
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals