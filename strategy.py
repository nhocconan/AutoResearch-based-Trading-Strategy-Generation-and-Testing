#!/usr/bin/env python3
"""
Experiment #121: 4h Primary + 1d/1w HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: Previous 4h strategies failed due to TOO MANY conflicting filters (Connors + Choppiness + 
Vol Spike + BB + HMA slope all needed to agree = 0 trades). This strategy SIMPLIFIES:

1. 1d HMA(21) for major trend bias (long only when bullish, short only when bearish)
2. 4h RSI(14) extremes for entry timing (<35 long, >65 short)
3. 4h Donchian(20) breakout confirmation (price near 20-day high/low)
4. 1w HMA(21) for ultra-long-term bias filter
5. ATR(14) trailing stoploss at 2.5x

Why this should work:
- Fewer filters = more trades (addresses #1 failure mode: 0 trades)
- 4h timeframe targets 20-50 trades/year (low fee drag)
- HTF (1d/1w) prevents counter-trend trades in strong moves
- Asymmetric sizing: 0.30 for trend-aligned, 0.20 for counter-trend mean revert
- Simple logic proven to work across multiple market regimes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (hh + ll) / 2.0
    return hh, ll, mid

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars only)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_hh, donchian_ll, donchian_mid = calculate_donchian(high, low, 20)
    
    # Price position in Donchian channel (0=low, 0.5=mid, 1=high)
    donchian_range = donchian_hh - donchian_ll
    donchian_range = np.where(donchian_range == 0, 1e-10, donchian_range)
    price_position = (close - donchian_ll) / donchian_range
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_hh[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W ULTRA-LONG TERM BIAS ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.1
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.1
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === DONCHIAN POSITION ===
        price_near_low = price_position[i] < 0.25
        price_near_high = price_position[i] > 0.75
        price_breakout_low = close[i] < donchian_ll[i] * 1.005
        price_breakout_high = close[i] > donchian_hh[i] * 0.995
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_confidence = 0
        
        # Path 1: 1d bullish + RSI oversold (trend pullback)
        if trend_1d_bullish and rsi_oversold:
            long_confidence += 2
        
        # Path 2: 1w bullish + RSI extreme low (deep pullback in bull)
        if trend_1w_bullish and rsi_extreme_low:
            long_confidence += 2
        
        # Path 3: Price above 1d HMA + RSI oversold + near Donchian low
        if price_above_1d_hma and rsi_oversold and price_near_low:
            long_confidence += 3
        
        # Path 4: 1d bullish + price near Donchian low (buying dip in uptrend)
        if trend_1d_bullish and price_near_low:
            long_confidence += 1
        
        # Path 5: RSI extreme low alone (capitulation long)
        if rsi_extreme_low:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence == 2 and bars_since_last_trade > 60:
            new_signal = REDUCED_SIZE
        elif long_confidence >= 1 and bars_since_last_trade > 100:
            new_signal = REDUCED_SIZE * 0.7
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: 1d bearish + RSI overbought (trend pullback)
        if trend_1d_bearish and rsi_overbought:
            short_confidence += 2
        
        # Path 2: 1w bearish + RSI extreme high (rally in bear)
        if trend_1w_bearish and rsi_extreme_high:
            short_confidence += 2
        
        # Path 3: Price below 1d HMA + RSI overbought + near Donchian high
        if price_below_1d_hma and rsi_overbought and price_near_high:
            short_confidence += 3
        
        # Path 4: 1d bearish + price near Donchian high (selling rally in downtrend)
        if trend_1d_bearish and price_near_high:
            short_confidence += 1
        
        # Path 5: RSI extreme high alone (euphoria short)
        if rsi_extreme_high:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence == 2 and bars_since_last_trade > 60:
            new_signal = -REDUCED_SIZE
        elif short_confidence >= 1 and bars_since_last_trade > 100:
            new_signal = -REDUCED_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~33 days on 4h) to ensure minimum trades
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 40:
                new_signal = REDUCED_SIZE * 0.5
            elif trend_1d_bearish and rsi_14[i] > 60:
                new_signal = -REDUCED_SIZE * 0.5
            elif rsi_14[i] < 30:
                new_signal = REDUCED_SIZE * 0.4
            elif rsi_14[i] > 70:
                new_signal = -REDUCED_SIZE * 0.4
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 55:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 45:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
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