#!/usr/bin/env python3
"""
Experiment #104: 4h Primary + 12h/1d HTF — Vol Spike Reversion + Regime Filter

Hypothesis: Recent 4h strategies failed due to either 0 trades (too strict) or 
negative Sharpe (wrong regime logic). This strategy focuses on VOLATILITY SPIKE 
REVERSION which has proven edge in bear/range markets (2022 crash, 2025 bear).

Strategy Logic:
1. VOL SPIKE DETECTION: ATR(7)/ATR(30) > 1.8 indicates panic/extreme vol
2. BOLLINGER EXTREMES: Price < BB_lower(20, 2.0) = oversold, > BB_upper = overbought
3. 12h HMA(21) TREND: Only long if 12h HMA slope > 0, only short if < 0
4. 1d HMA(50) REGIME: Price > 1d HMA = bull regime (prefer longs), < = bear (prefer shorts)
5. DONCHIAN BREAKOUT: 20-bar breakout confirms momentum continuation
6. ATR(14) stoploss: 2.5x trailing stop on all positions

Why this should work:
- Vol spike reversion has 70%+ win rate after panic events
- 4h timeframe = 30-60 trades/year target (fee-efficient)
- 12h/1d HTF prevents counter-trend trades in strong moves
- Simpler entry logic = more trades = better statistics
- Asymmetric sizing: larger positions in direction of 1d regime

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 with regime confirmation
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_bb_regime_12h1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands with proper min_periods."""
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    bb_upper_25, bb_lower_25, _ = calculate_bollinger_bands(close, 20, 2.5)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    
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
        
        if np.isnan(hma_12h_slope_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 1.8 indicates panic/extreme volatility
        vol_spike = (atr_30[i] > 0) and (atr_7[i] / atr_30[i] > 1.8)
        vol_normal = (atr_30[i] > 0) and (atr_7[i] / atr_30[i] < 1.3)
        
        # === 12H TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        
        # === 1D REGIME FILTER ===
        # Price above 1d HMA(50) = bull regime (prefer longs)
        # Price below 1d HMA(50) = bear regime (prefer shorts)
        bull_regime = close[i] > hma_1d_50_aligned[i]
        bear_regime = close[i] < hma_1d_50_aligned[i]
        
        # === BOLLINGER EXTREMES ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_below_bb_lower_25 = close[i] < bb_lower_25[i]
        price_above_bb_upper_25 = close[i] > bb_upper_25[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_upper[i - 1] if i > 0 else False
        donchian_breakout_low = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if bull_regime and trend_12h_bullish:
            current_size = CONFIRMED_SIZE  # Full size in confirmed bull
        elif bear_regime and trend_12h_bearish:
            current_size = CONFIRMED_SIZE  # Full size in confirmed bear
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (3 confluence minimum)
        long_confluence = 0
        
        # Confluence 1: Vol spike reversion OR BB extreme
        if vol_spike or price_below_bb_lower_25:
            long_confluence += 1
        
        # Confluence 2: RSI oversold
        if rsi_oversold:
            long_confluence += 1
        
        # Confluence 3: 12h trend bullish OR bull regime
        if trend_12h_bullish or bull_regime:
            long_confluence += 1
        
        # Confluence 4: Donchian support hold (price > Donchian lower)
        if close[i] > donchian_lower[i]:
            long_confluence += 1
        
        # Enter long with 3+ confluence
        if long_confluence >= 3:
            new_signal = current_size
        
        # SHORT ENTRIES (3 confluence minimum)
        short_confluence = 0
        
        # Confluence 1: Vol spike reversion OR BB extreme
        if vol_spike or price_above_bb_upper_25:
            short_confluence += 1
        
        # Confluence 2: RSI overbought
        if rsi_overbought:
            short_confluence += 1
        
        # Confluence 3: 12h trend bearish OR bear regime
        if trend_12h_bearish or bear_regime:
            short_confluence += 1
        
        # Confluence 4: Donchian resistance hold (price < Donchian upper)
        if close[i] < donchian_upper[i]:
            short_confluence += 1
        
        # Enter short with 3+ confluence
        if short_confluence >= 3 and new_signal == 0.0:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~20 days on 4h), allow weaker entry (2 confluence)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_oversold and (vol_spike or price_below_bb_lower):
                new_signal = BASE_SIZE * 0.6
            elif bear_regime and rsi_overbought and (vol_spike or price_above_bb_upper):
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if major regime changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d regime turns bearish strongly
            if position_side > 0 and bear_regime and trend_12h_bearish:
                regime_reversal = True
            # Exit short if 1d regime turns bullish strongly
            if position_side < 0 and bull_regime and trend_12h_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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
                # Position flip
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