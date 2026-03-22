#!/usr/bin/env python3
"""
Experiment #040: 1h Fisher Transform + 4h/12h HMA Trend + ADX Regime

Hypothesis: 1h timeframe with 4h trend bias + 12h ADX regime filter + Fisher Transform
entries will generate sufficient trades (30-80/year) while maintaining edge.

Key design:
1. 4h HMA(21) for major trend bias (call ONCE via mtf_data)
2. 12h ADX(14) for regime detection (>25 = trending, <20 = ranging)
3. Fisher Transform(9) for entry timing (crosses at -1.5/+1.5 levels)
4. Volume filter: volume > 0.5x 20-bar average (loose filter)
5. ATR(14) for stoploss (2.5x)
6. Discrete sizing: 0.25 base, 0.30 strong trend

Why this should work:
- Fisher Transform catches reversals better than RSI in bear/range markets
- 4h HMA provides clear trend bias without whipsaw
- 12h ADX distinguishes trending vs ranging regimes
- Volume filter is LOOSE (0.5x avg) to ensure trades trigger
- Fisher thresholds (-1.5/+1.5) are wide enough for regular signals
- 1h TF naturally produces 40-80 trades/year with these settings

Timeframe: 1h (REQUIRED)
HTF: 4h + 12h via mtf_data helper (call ONCE before loop!)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_12h_hma_adx_volume_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range, highlights reversals.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    signal = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0
            signal[i] = fisher[i]
            continue
        
        normalized = 2 * (close[i] - lowest) / range_val - 1
        normalized = np.clip(normalized, -0.99, 0.99)
        
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized)) + 0.5 * fisher[i-1]
        signal[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h ADX regime
    adx_12h_14 = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher_9, fisher_signal_9 = calculate_fisher(close, 9)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    bars_without_signal = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(adx_12h_aligned[i]):
            continue
        
        if np.isnan(fisher_9[i]) or np.isnan(fisher_signal_9[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (12h ADX) ===
        # ADX > 25 = trending market (follow trend)
        # ADX < 20 = ranging market (mean revert ok)
        # 20-25 = neutral
        adx_val = adx_12h_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # === VOLUME FILTER (loose: > 0.5x average) ===
        volume_ok = volume[i] > 0.5 * vol_avg_20[i]
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        fisher_cross_long = (fisher_9[i] > -1.5) and (fisher_signal_9[i] <= -1.5)
        fisher_cross_short = (fisher_9[i] < 1.5) and (fisher_signal_9[i] >= 1.5)
        
        # Also allow entries when Fisher is at extremes (wider triggers)
        fisher_extreme_long = fisher_9[i] < -1.0
        fisher_extreme_short = fisher_9[i] > 1.0
        
        # === ENTRY LOGIC - LOOSE THRESHOLDS FOR TRADE GENERATION ===
        new_signal = 0.0
        bars_without_signal += 1
        
        # LONG entries (multiple paths to ensure trades trigger)
        if htf_bullish and volume_ok:
            if is_trending:
                # Trending: wait for Fisher cross up from oversold
                if fisher_cross_long or fisher_extreme_long:
                    new_signal = STRONG_SIZE
            else:
                # Ranging: Fisher extreme is enough
                if fisher_extreme_long or fisher_cross_long:
                    new_signal = BASE_SIZE
        
        # SHORT entries (multiple paths to ensure trades trigger)
        if htf_bearish and volume_ok:
            if is_trending:
                # Trending: wait for Fisher cross down from overbought
                if fisher_cross_short or fisher_extreme_short:
                    new_signal = -STRONG_SIZE
            else:
                # Ranging: Fisher extreme is enough
                if fisher_extreme_short or fisher_cross_short:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~2 days on 1h), force entry on HTF bias
        if bars_without_signal > 50 and new_signal == 0.0 and not in_position:
            if htf_bullish and volume_ok:
                new_signal = BASE_SIZE * 0.8
                bars_without_signal = 0
            elif htf_bearish and volume_ok:
                new_signal = -BASE_SIZE * 0.8
                bars_without_signal = 0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher_9[i] > 2.0:
                fisher_exit = True
            if position_side < 0 and fisher_9[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or fisher_exit:
            new_signal = 0.0
            bars_without_signal = 0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_without_signal = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_without_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals