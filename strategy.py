#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d regime filter (ADX < 20 = range, ADX > 25 = trend)
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h data)
# - Regime filter: 1d ADX(14) > 25 = trending (follow Elder Ray signals), < 20 = ranging (fade Elder Ray extremes)
# - In trending regime: Long when Bull Power crosses above 0 with rising Bear Power (bullish momentum)
#   Short when Bear Power crosses below 0 with falling Bull Power (bearish momentum)
# - In ranging regime: Long when Bear Power < -0.5 * ATR(10) (oversold), Short when Bull Power > 0.5 * ATR(10) (overbought)
# - Volume confirmation: current 6h volume > 1.2x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)

name = "6h_1d_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher = stronger bulls
    bear_power = low - ema_13   # Lower (more negative) = stronger bears
    
    # 6h ATR(10) for ranging regime thresholds
    def calculate_atr(high, low, close, period=10):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_10 = calculate_atr(high, low, close, 10)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_10[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Regime filter: 1d ADX > 25 = trending, < 20 = ranging
        trending_regime = adx_1d_aligned[i] > 25
        ranging_regime = adx_1d_aligned[i] < 20
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if trending_regime:
                # Exit long when Bear Power crosses above 0 (bulls weakening)
                if bear_power[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # ranging regime
                # Exit long when price returns to mean (Bear Power > -0.2 * ATR)
                if bear_power[i] > -0.2 * atr_10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if trending_regime:
                # Exit short when Bull Power crosses below 0 (bears weakening)
                if bull_power[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # ranging regime
                # Exit short when price returns to mean (Bull Power < 0.2 * ATR)
                if bull_power[i] < 0.2 * atr_10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            if volume_confirmed:
                if trending_regime:
                    # Trending: follow Elder Ray momentum
                    # Long: Bull Power crosses above 0 with Bear Power still negative (bulls taking control)
                    if bull_power[i] > 0 and bull_power[i-1] <= 0 and bear_power[i] < 0:
                        position = 1
                        signals[i] = position_size
                    # Short: Bear Power crosses below 0 with Bull Power still positive (bears taking control)
                    elif bear_power[i] < 0 and bear_power[i-1] >= 0 and bull_power[i] > 0:
                        position = -1
                        signals[i] = -position_size
                else:  # ranging regime
                    # Ranging: fade Elder Ray extremes
                    # Long: Bear Power extended below -0.5 * ATR (oversold)
                    if bear_power[i] < -0.5 * atr_10[i]:
                        position = 1
                        signals[i] = position_size
                    # Short: Bull Power extended above 0.5 * ATR (overbought)
                    elif bull_power[i] > 0.5 * atr_10[i]:
                        position = -1
                        signals[i] = -position_size
    
    return signals