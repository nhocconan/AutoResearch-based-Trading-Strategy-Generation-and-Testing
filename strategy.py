#!/usr/bin/env python3
"""
Experiment #001: 4h Dual-Regime Strategy with 1d Trend Filter

Hypothesis: Markets alternate between trending and ranging regimes. A single strategy
cannot excel in both. This implements regime-adaptive logic:
- CHOPPINESS INDEX > 61.8 = range regime → mean reversion (RSI extremes + BB)
- CHOPPINESS INDEX < 38.2 = trend regime → trend following (HMA + Donchian breakout)
- 1d HMA(21) provides major trend bias (only trade in direction of daily trend)

Why 4h works:
- Natural 20-50 trades/year (fee drag manageable at 0.05% RT)
- Filters 15m/1h noise while capturing meaningful moves
- 1d HTF filter prevents counter-trend trades that destroy Sharpe in bear markets

Key components:
1. Choppiness Index (14) for regime detection
2. HMA(16/48) crossover for trend confirmation
3. Donchian(20) breakout for trend entries
4. RSI(14) + Bollinger(20,2) for mean reversion entries
5. 1d HMA(21) for major trend bias via mtf_data helper
6. ATR(14) trailing stoploss at 2.5x

Position sizing: 0.25-0.30 discrete levels
Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_hma_1d_filter_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=period, min_periods=period).sum()
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1D TREND BIAS (major direction filter) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER MEAN REVERSION ===
        bb_long = close[i] < bb_lower[i] and not np.isnan(bb_lower[i])
        bb_short = close[i] > bb_upper[i] and not np.isnan(bb_upper[i])
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND REGIME: Follow breakouts in trend direction
        if is_trending:
            # Long: HMA bullish + breakout + daily bias confirms
            if hma_bullish and breakout_long and daily_bullish:
                new_signal = BASE_SIZE
            
            # Short: HMA bearish + breakout + daily bias confirms
            elif hma_bearish and breakout_short and daily_bearish:
                new_signal = -BASE_SIZE
        
        # RANGE REGIME: Mean reversion at Bollinger extremes
        elif is_ranging:
            # Long: RSI oversold + price below BB lower + daily not strongly bearish
            if rsi_oversold and bb_long and not daily_bearish:
                new_signal = BASE_SIZE
            
            # Short: RSI overbought + price above BB upper + daily not strongly bullish
            elif rsi_overbought and bb_short and not daily_bullish:
                new_signal = -BASE_SIZE
        
        # NEUTRAL REGIME (38.2 < CHOP < 61.8): Use HMA crossover only
        else:
            if hma_bullish and daily_bullish:
                new_signal = BASE_SIZE * 0.7
            elif hma_bearish and daily_bearish:
                new_signal = -BASE_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if hma_bullish and daily_bullish and rsi_14[i] < 60:
                new_signal = BASE_SIZE * 0.6
            elif hma_bearish and daily_bearish and rsi_14[i] > 40:
                new_signal = -BASE_SIZE * 0.6
        
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
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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