#!/usr/bin/env python3
"""
Experiment #119: 4h Primary + 1d HTF — Dual Regime Breakout + Mean Reversion

Hypothesis: Previous 4h strategies failed because they were either too strict (0 trades)
or too trend-focused (negative Sharpe in bear markets). This strategy combines:

1. DONCHIAN BREAKOUT (trend mode): 20-period high/low breakouts for trending markets
2. RSI MEAN REVERSION (range mode): RSI(14) extremes for choppy markets
3. CHOPPINESS INDEX: Regime switch (CHOP>55 = range, CHOP<45 = trend)
4. 1d HMA(21) SLOPE: Major trend bias for position sizing asymmetry
5. ATR TRAILING STOP: 2.5*ATR(14) to protect capital

Why this should work:
- Dual regime adapts to market conditions (trend vs range)
- 4h timeframe = 20-50 trades/year target (low fee drag)
- 1d HTF prevents fighting major trends
- Asymmetric sizing: larger positions with 1d trend, smaller against
- Looser entry conditions than previous failed experiments

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_donchian_rsi_1d_v1"
timeframe = "4h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Price position within Donchian channel (0=lower, 1=upper, 0.5=middle)
    donchian_range = donchian_upper - donchian_lower
    donchian_range = np.where(donchian_range == 0, 1e-10, donchian_range)
    price_in_donchian = (close - donchian_lower) / donchian_range
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === PRICE IN CHANNEL ===
        price_near_lower = price_in_donchian[i] < 0.2
        price_near_upper = price_in_donchian[i] > 0.8
        
        # === POSITION SIZING BASED ON 1D TREND ===
        if trend_1d_bullish:
            long_size = BASE_SIZE
            short_size = REDUCED_SIZE
        elif trend_1d_bearish:
            long_size = REDUCED_SIZE
            short_size = BASE_SIZE
        else:
            long_size = REDUCED_SIZE
            short_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === LONG ENTRIES (Multiple paths for more trades) ===
        long_triggered = False
        long_strength = 0
        
        # Path 1: Trend mode + Donchian breakout + 1d bullish bias
        if is_trend_market and breakout_long and trend_1d_bullish:
            long_triggered = True
            long_strength = 3
        
        # Path 2: Trend mode + Donchian breakout (any 1d bias)
        if is_trend_market and breakout_long and not long_triggered:
            long_triggered = True
            long_strength = 2
        
        # Path 3: Range mode + RSI oversold + price near lower
        if is_range_market and rsi_oversold and price_near_lower and not long_triggered:
            long_triggered = True
            long_strength = 2
        
        # Path 4: Range mode + RSI extreme low
        if is_range_market and rsi_extreme_low and not long_triggered:
            long_triggered = True
            long_strength = 2
        
        # Path 5: Price below 1d HMA + RSI oversold (pullback in bull)
        if price_above_1d_hma and rsi_oversold and not long_triggered:
            long_triggered = True
            long_strength = 1
        
        # Path 6: Simple RSI oversold (fallback for more trades)
        if rsi_14[i] < 30 and not long_triggered:
            long_triggered = True
            long_strength = 1
        
        if long_triggered:
            if long_strength >= 2:
                new_signal = long_size
            elif long_strength == 1 and bars_since_last_trade > 40:
                new_signal = long_size * 0.6
        
        # === SHORT ENTRIES ===
        short_triggered = False
        short_strength = 0
        
        # Path 1: Trend mode + Donchian breakdown + 1d bearish bias
        if is_trend_market and breakout_short and trend_1d_bearish:
            short_triggered = True
            short_strength = 3
        
        # Path 2: Trend mode + Donchian breakdown (any 1d bias)
        if is_trend_market and breakout_short and not short_triggered:
            short_triggered = True
            short_strength = 2
        
        # Path 3: Range mode + RSI overbought + price near upper
        if is_range_market and rsi_overbought and price_near_upper and not short_triggered:
            short_triggered = True
            short_strength = 2
        
        # Path 4: Range mode + RSI extreme high
        if is_range_market and rsi_extreme_high and not short_triggered:
            short_triggered = True
            short_strength = 2
        
        # Path 5: Price below 1d HMA + RSI overbought (rally in bear)
        if price_below_1d_hma and rsi_overbought and not short_triggered:
            short_triggered = True
            short_strength = 1
        
        # Path 6: Simple RSI overbought (fallback)
        if rsi_14[i] > 70 and not short_triggered:
            short_triggered = True
            short_strength = 1
        
        if short_triggered:
            if short_strength >= 2:
                new_signal = -short_size
            elif short_strength == 1 and bars_since_last_trade > 40:
                new_signal = -short_size * 0.6
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 40:
                new_signal = long_size * 0.4
            elif trend_1d_bearish and rsi_14[i] > 60:
                new_signal = -short_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = long_size * 0.3
            elif rsi_14[i] > 70:
                new_signal = -short_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trend_market and trend_1d_bearish and rsi_14[i] > 60:
                regime_reversal = True
            if position_side < 0 and is_trend_market and trend_1d_bullish and rsi_14[i] < 40:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
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