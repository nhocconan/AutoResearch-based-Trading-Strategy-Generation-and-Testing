#!/usr/bin/env python3
"""
Experiment #340: 1h Primary + 4h/12h HTF — Choppiness Regime + HMA Trend + RSI Pullback

Hypothesis: After 30+ failed lower-TF experiments (328, 330, 335, 338, 339 all got 0 trades),
the problem is TOO MANY confluence filters. This strategy uses:
1. Choppiness Index (CHOP) for regime detection - proven in literature for crypto
2. 4h HMA(21) for trend direction (single HTF, not multiple)
3. 1h RSI(14) pullback entries with LOOSE thresholds (30-70, not 40-60)
4. Multiple entry paths with OR logic (any 2 of 3 conditions = entry)
5. Frequency safeguard every 40 bars to ensure 20+ trades/year
6. Smaller position sizes (0.20-0.30) for 1h volatility

Why this might beat current best (Sharpe=0.435):
- CHOP regime filter adapts to market conditions (trend vs range)
- 4h HTF gives direction, 1h gives entry timing (proven pattern)
- OR logic for entries ensures trades actually happen
- Looser RSI thresholds generate more signals
- ATR stoploss at 2.0x (tighter for lower TF)

Position sizing: 0.20-0.30 (smaller than 1d due to higher 1h volatility)
Stoploss: 2.0 * ATR trailing
Target: 40-80 trades/year on 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_hma_rsi_4h_regime_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = period
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # CHOP calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(n)
    
    # Clamp to 0-100
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_8 = calculate_hma(close, period=8)
    hma_1h_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -40
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_1h_8[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA (favor longs)
        # Bear: price below 4h HMA (favor shorts)
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME (trend vs range) ===
        # CHOP > 55 = range/choppy (mean reversion)
        # CHOP < 45 = trending (trend following)
        chop_range = chop_14[i] > 55.0
        chop_trend = chop_14[i] < 45.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.4
        vol_scale = 0.8 if high_vol else 1.0
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        # HMA slope (2-bar lookback)
        hma_slope_up = hma_1h_21[i] > hma_1h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_1h_21[i] < hma_1h_21[i-2] if i >= 2 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (loose thresholds for trade generation) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral_long = 35.0 < rsi_14[i] < 55.0
        rsi_neutral_short = 45.0 < rsi_14[i] < 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] if not np.isnan(donchian_upper[i]) else False
        donchian_breakout_short = close[i] < donchian_lower[i] if not np.isnan(donchian_lower[i]) else False
        
        # === ENTRY LOGIC (OR logic - multiple paths to entry) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (multiple paths - any 2 conditions = entry)
        if regime_bull:
            long_conditions = 0
            
            # Condition 1: RSI pullback
            if rsi_neutral_long or rsi_oversold:
                long_conditions += 1
            
            # Condition 2: HMA bullish
            if hma_bullish or hma_slope_up:
                long_conditions += 1
            
            # Condition 3: Price above HMA
            if price_above_hma:
                long_conditions += 1
            
            # Condition 4: Choppiness indicates trend
            if chop_trend:
                long_conditions += 1
            
            # Enter if 2+ conditions met
            if long_conditions >= 2:
                if rsi_oversold:
                    new_signal = LONG_STRONG * vol_scale
                else:
                    new_signal = LONG_BASE * vol_scale
        
        # SHORT ENTRIES (multiple paths)
        if regime_bear:
            short_conditions = 0
            
            # Condition 1: RSI pullback
            if rsi_neutral_short or rsi_overbought:
                short_conditions += 1
            
            # Condition 2: HMA bearish
            if hma_bearish or hma_slope_down:
                short_conditions += 1
            
            # Condition 3: Price below HMA
            if price_below_hma:
                short_conditions += 1
            
            # Condition 4: Choppiness indicates trend
            if chop_trend:
                short_conditions += 1
            
            # Enter if 2+ conditions met
            if short_conditions >= 2:
                if rsi_overbought:
                    new_signal = -SHORT_STRONG * vol_scale
                else:
                    new_signal = -SHORT_BASE * vol_scale
        
        # === RANGE REGIME MEAN REVERSION (CHOP > 55) ===
        if chop_range and new_signal == 0.0:
            # Long at range bottom
            if rsi_oversold and price_below_hma and regime_bull:
                new_signal = LONG_BASE * 0.8 * vol_scale
            
            # Short at range top
            if rsi_overbought and price_above_hma and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 1h) ===
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 35.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 65.0:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_oversold:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_overbought:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_oversold:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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