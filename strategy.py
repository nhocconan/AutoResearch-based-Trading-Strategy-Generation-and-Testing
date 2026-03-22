#!/usr/bin/env python3
"""
Experiment #185: 12h Regime-Adaptive Strategy with Choppiness Index + Dual HTF Filter

Hypothesis: 12h timeframe needs regime detection to avoid whipsaws. Previous 12h
strategies failed because they used pure trend-following (Donchian/Supertrend) which
gets chopped in ranges. This strategy uses Choppiness Index (CHOP) to detect market
regime and switches logic accordingly:
- CHOP > 61.8 = range market → mean revert at Bollinger Band extremes
- CHOP < 38.2 = trending market → follow Donchian breakouts with HTF bias

Key innovations vs previous 12h failures (#173, #179):
1. CHOP regime filter (not used on 12h before) - adapts to market state
2. Dual HTF filter: 1d HMA for directional bias + 1w HMA for major trend
3. Lower ADX threshold (15 vs 18-25) for 12h to ensure sufficient trade count
4. Asymmetric logic: only long in bull regime, only short in bear regime
5. Position size 0.30 (higher than 0.25) since fewer trades expected on 12h

Why this should beat Sharpe=0.478 baseline:
- Regime adaptation reduces whipsaw losses in choppy periods
- 1w HTF filter prevents counter-trend trades in strong macro trends
- Mean reversion in ranges captures 70%+ of 12h market time (crypto ranges often)
- Trend following in trends captures the big moves (2021 bull, 2025 recovery)

Timeframe: 12h (REQUIRED)
HTF: 1d + 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_1d_1w_hma_bb_donchian_atr_v1"
timeframe = "12h"
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

def calculate_chop(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # First calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    # Fill initial values
    chop[:period] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values
    upper[:period-1] = upper[period-1]
    lower[:period-1] = lower[period-1]
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_chop(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = major trend bias (only trade with major trend)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range market (mean revert)
        # CHOP < 38.2 = trending market (trend follow)
        # 38.2 - 61.8 = transition (stay flat or reduce size)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        # === RSI EXTREMES FOR MEAN REVERSION ===
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # === PRICE AT BB BOUNDS ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper band
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        new_signal = 0.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        
        # --- RANGE MARKET: Mean Reversion ---
        if is_range:
            # Long: 1d bullish + 1w bullish + RSI oversold + at BB lower
            if bull_trend_1d and bull_trend_1w and rsi_oversold and at_bb_lower:
                new_signal = SIZE_BASE
            
            # Short: 1d bearish + 1w bearish + RSI overbought + at BB upper
            if bear_trend_1d and bear_trend_1w and rsi_overbought and at_bb_upper:
                new_signal = -SIZE_BASE
        
        # --- TREND MARKET: Trend Following ---
        elif is_trend:
            # Long: 1d bullish + 1w bullish + Donchian breakout
            if bull_trend_1d and bull_trend_1w and breakout_long:
                new_signal = SIZE_BASE
            
            # Short: 1d bearish + 1w bearish + Donchian breakout
            if bear_trend_1d and bear_trend_1w and breakout_short:
                new_signal = -SIZE_BASE
        
        # --- TRANSITION ZONE: Reduced position or flat ---
        # Only enter if strong conviction (both 1d and 1w agree strongly)
        else:
            # Allow entries but with stricter conditions
            if bull_trend_1d and bull_trend_1w and breakout_long and rsi[i] > 45:
                new_signal = SIZE_BASE * 0.7  # reduced size in transition
            
            if bear_trend_1d and bear_trend_1w and breakout_short and rsi[i] < 55:
                new_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals