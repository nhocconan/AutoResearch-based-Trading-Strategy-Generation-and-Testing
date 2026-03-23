#!/usr/bin/env python3
"""
Experiment #029: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX + RSI Pullback

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA.
Combined with ADX trend strength filter and RSI pullback entries, this should capture
trending moves while avoiding choppy whipsaws. Different from CRSI+Choppiness approach.

Key innovations:
1. KAMA (Efficiency Ratio adaptive): Smooths in chop, follows in trends
2. ADX(14) > 20 filter: Only trade when trend has strength
3. RSI(14) pullback: Enter on dips in uptrend (RSI 35-50), rallies in downtrend (RSI 50-65)
4. Donchian(20) breakout confirmation: Price must break recent high/low
5. 1d HMA macro bias: Align with daily trend direction

Why this differs from #024:
- #024 uses CRSI mean-reversion + Choppiness regime switch
- This uses KAMA trend-following + ADX strength + RSI pullback timing
- Captures different market regimes (trending vs ranging)

Entry conditions (loose enough for trades):
- Long: KAMA bullish + ADX > 18 + RSI 35-55 + price > Donchian mid + 1d HMA bullish
- Short: KAMA bearish + ADX > 18 + RSI 45-65 + price < Donchian mid + 1d HMA bearish

Position size: 0.28 (discrete)
Stoploss: 2.5 * ATR trailing
Target trades: 30-50/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_donchian_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over er_period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    vol_sum = np.zeros(n)
    for i in range(er_period, n):
        vol_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (0 = noise, 1 = pure trend)
    er = np.zeros(n)
    for i in range(er_period, n):
        if vol_sum[i] > 0:
            er[i] = price_change[i] / vol_sum[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = np.zeros(n)
    for i in range(er_period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction).
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    # KAMA slope (trend direction)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = kama_4h[i] - kama_4h[i-5]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_mid[i]) or np.isnan(kama_slope[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = kama_slope[i] > 0 and close[i] > kama_4h[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama_4h[i]
        
        # === ADX TREND STRENGTH ===
        # Lower threshold (18 instead of 20) for more trades
        trend_strong = adx_14[i] > 18.0
        
        # === RSI PULLBACK ZONES ===
        # Long pullback: RSI 35-55 (dip in uptrend)
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 55.0
        # Short pullback: RSI 45-65 (rally in downtrend)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 65.0
        
        # === DONCHIAN CONFIRMATION ===
        price_above_donchian_mid = close[i] > donchian_mid[i]
        price_below_donchian_mid = close[i] < donchian_mid[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: KAMA bullish + ADX strong + RSI pullback + Donchian + 1d bias
        if kama_bullish and trend_strong and rsi_long_pullback:
            if price_above_donchian_mid and price_above_hma_1d:
                new_signal = POSITION_SIZE
            elif price_above_donchian_mid:
                # Weaker signal without 1d confirmation
                new_signal = POSITION_SIZE * 0.7
        
        # Short entry: KAMA bearish + ADX strong + RSI pullback + Donchian + 1d bias
        elif kama_bearish and trend_strong and rsi_short_pullback:
            if price_below_donchian_mid and price_below_hma_1d:
                new_signal = -POSITION_SIZE
            elif price_below_donchian_mid:
                # Weaker signal without 1d confirmation
                new_signal = -POSITION_SIZE * 0.7
        
        # === HOLD POSITION LOGIC ===
        # Keep position if we're already in one and no exit signal
        if in_position and new_signal == 0.0:
            # Check if trend is still valid before holding
            if position_side > 0:
                # Hold long if KAMA still bullish or RSI not overbought
                if kama_bullish or rsi_14[i] < 70:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if KAMA still bearish or RSI not oversold
                if kama_bearish or rsi_14[i] > 30:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if KAMA turns bearish + ADX still strong
        if in_position and position_side > 0:
            if kama_bearish and trend_strong:
                new_signal = 0.0
        
        # Exit short if KAMA turns bullish + ADX still strong
        if in_position and position_side < 0:
            if kama_bullish and trend_strong:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREMES ===
        # Exit long if RSI very overbought
        if in_position and position_side > 0:
            if rsi_14[i] > 75:
                new_signal = 0.0
        
        # Exit short if RSI very oversold
        if in_position and position_side < 0:
            if rsi_14[i] < 25:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals