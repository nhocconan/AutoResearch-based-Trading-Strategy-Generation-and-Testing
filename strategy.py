#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 13-period EMA + Regime Filter
# Elder Ray = Bull Power = High - EMA13, Bear Power = Low - EMA13
# Regime: ADX > 25 = trending (follow Elder Ray), ADX < 20 = ranging (fade extremes)
# In trending markets: Buy when Bull Power > 0 and rising, Sell when Bear Power < 0 and falling
# In ranging markets: Fade when Bull Power > 0.5*ATR (overbought) or Bear Power < -0.5*ATR (oversold)
# Uses 13-period for responsiveness, avoids overtrading with regime filter
# Works in bull/bear: trends captured in trending regimes, mean reversion in ranging

name = "6h_13_ElderRay_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for weekly context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Weekly Trend Filter (EMA50 vs EMA200 on 1d) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    # Uptrend on daily = long bias, downtrond = short bias
    daily_uptrend = ema50_1d > ema200_1d
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    
    # === 6h Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Strength of bulls
    bear_power = low - ema13   # Strength of bears (negative values)
    
    # ATR(20) for volatility normalization and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for regime detection
    # +DM, -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed +DM, -DM, TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI, -DI, DX
    plus_di = 100 * plus_dm_smooth / np.where(atr14 > 0, atr14, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(atr14 > 0, atr14, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, np.nan)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr20[i]) or np.isnan(adx[i]) or 
            np.isnan(daily_uptrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # Get values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr20[i]
        daily_up = daily_uptrend_aligned[i] > 0.5
        
        if position == 0:
            # ENTRY LOGIC
            if is_trending:
                # In trending market: follow Elder Ray with daily bias
                if bull_val > 0 and daily_up:  # Bullish power + daily uptrend
                    signals[i] = 0.25
                    position = 1
                elif bear_val < 0 and not daily_up:  # Bearish power + daily downtrend
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging or transition
                # Fade extremes: overbought/oversold conditions
                if bull_val > 0.5 * atr_val:  # Overbought
                    signals[i] = -0.25
                    position = -1
                elif bear_val < -0.5 * atr_val:  # Oversold
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # LONG EXIT
            if is_trending:
                # Exit trend when power fades or daily bias changes
                if bull_val <= 0 or not daily_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit when power normalizes or opposite extreme
                if bull_val < 0.2 * atr_val or bear_val > -0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # SHORT EXIT
            if is_trending:
                # Exit trend when power fades or daily bias changes
                if bear_val >= 0 or daily_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit when power normalizes or opposite extreme
                if bear_val > -0.2 * atr_val or bull_val < 0.2 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals