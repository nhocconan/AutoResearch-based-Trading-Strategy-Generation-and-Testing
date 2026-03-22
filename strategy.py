#!/usr/bin/env python3
"""
Experiment #277: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 12 consecutive failures with complex multi-filter strategies,
simplify to proven components that actually generate trades:
1. HMA(21) on 1d for PRIMARY trend direction (proven in #251, #276)
2. HMA(21) on 1w for MEGATREND filter (only trade with weekly trend)
3. RSI(14) on 1d for entry timing (relaxed thresholds: 35/65 not 25/75)
4. ATR(14) trailing stop for risk management (2.5x ATR)
5. MINIMAL filters to ensure 20-50 trades/year (NOT 0 trades like #265, #268, #270, #275)

Key differences from failed #274 (Sharpe=-0.634):
- SIMPLER logic: fewer confluence requirements
- RELAXED RSI: 35/65 thresholds (not 25/75) to generate more trades
- Weekly HMA for megatrend (not 12h) — cleaner signal
- NO Donchian, NO Choppiness, NO ADX — these added complexity without value
- Force entries when conditions met (no artificial frequency limits)

Position sizing: 0.25 base, 0.30 strong (discrete, max 0.35)
Target: 25-40 trades/year on 1d (appropriate for daily timeframe)
Stoploss: 2.5 * ATR trailing (mandatory per Rule 6)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_simple_1w_v1"
timeframe = "1d"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (megatrend filter)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W MEGATREND (only trade with weekly trend) ===
        # Bull megatrend: price above 1w HMA
        # Bear megatrend: price below 1w HMA
        megatrend_bull = close[i] > hma_1w_21_aligned[i]
        megatrend_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND (primary direction) ===
        trend_bull = close[i] > hma_1d_21[i]
        trend_bear = close[i] < hma_1d_21[i]
        hma_bullish_cross = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish_cross = hma_1d_21[i] < hma_1d_50[i]
        
        # === RSI ENTRY SIGNALS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 35.0
        rsi_extreme_overbought = rsi_14[i] > 65.0
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: Megatrend bull + Daily trend bull + RSI pullback (not extreme)
        if megatrend_bull and trend_bull and rsi_oversold and not rsi_extreme_oversold:
            new_signal = BASE_SIZE
        
        # LONG STRONG: Megatrend bull + Daily trend bull + RSI extreme oversold
        if megatrend_bull and trend_bull and rsi_extreme_oversold:
            new_signal = STRONG_SIZE
        
        # LONG: Megatrend bull + HMA bullish cross + RSI neutral (momentum entry)
        if megatrend_bull and hma_bullish_cross and rsi_neutral:
            if new_signal == 0.0:
                new_signal = BASE_SIZE
        
        # SHORT: Megatrend bear + Daily trend bear + RSI rally (not extreme)
        if megatrend_bear and trend_bear and rsi_overbought and not rsi_extreme_overbought:
            if new_signal == 0.0 or abs(new_signal) < BASE_SIZE:
                new_signal = -BASE_SIZE
        
        # SHORT STRONG: Megatrend bear + Daily trend bear + RSI extreme overbought
        if megatrend_bear and trend_bear and rsi_extreme_overbought:
            new_signal = -STRONG_SIZE
        
        # SHORT: Megatrend bear + HMA bearish cross + RSI neutral (momentum entry)
        if megatrend_bear and hma_bearish_cross and rsi_neutral:
            if new_signal == 0.0:
                new_signal = -BASE_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily trend turns bearish
            if position_side > 0 and trend_bear:
                trend_reversal = True
            # Short position but daily trend turns bullish
            if position_side < 0 and trend_bull:
                trend_reversal = True
        
        # === MEGATREND REVERSAL EXIT (stronger signal) ===
        megatrend_reversal = False
        if in_position and position_side != 0:
            # Long but megatrend turns bear
            if position_side > 0 and megatrend_bear:
                megatrend_reversal = True
            # Short but megatrend turns bull
            if position_side < 0 and megatrend_bull:
                megatrend_reversal = True
        
        if stoploss_triggered or trend_reversal or megatrend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals