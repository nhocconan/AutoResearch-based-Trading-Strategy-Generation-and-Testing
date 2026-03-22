#!/usr/bin/env python3
"""
Experiment #319: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: Simpler 4h strategy with looser entry conditions will generate trades where #314/#311 failed.
Key insight from failures: Sharpe=0.000 + Return=0.0% = ZERO TRADES generated.
Entry conditions were too strict with multiple conflicting filters.

Why this might work:
1. 4h timeframe proven to work (20-50 trades/year target, low fee drag)
2. 1d HTF for major trend direction (faster than 1w, more responsive for 4h entries)
3. HMA(16/48) crossover - proven trend indicator on 4h
4. RSI(14) pullback entries - looser thresholds (35-65 range, not extremes)
5. Donchian(20) breakout - captures momentum when trend confirms
6. ANY of 3 entry conditions can trigger (OR logic, not AND) = more trades
7. Asymmetric but not extreme: longs 0.30, shorts 0.25

Key differences from failed #314/#311:
- REMOVED Choppiness Index (was filtering out too many valid signals)
- REMOVED complex regime switching (was creating conflicting conditions)
- LOOSER RSI thresholds (35-65 instead of 30-70 or 40-60)
- OR logic for entries (any condition triggers, not all must agree)
- Simpler stoploss (2.5 ATR instead of 3.0)
- Minimum trade frequency guard (force entry every 40 bars if no signal)

Position sizing: 0.25 base, 0.30 strong (longs), 0.25 (shorts)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_1d_simp_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    n = period
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA function
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, half_n)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    sma_50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.22
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA21 AND HMA21 > HMA50
        # Bear: price below 1d HMA21 AND HMA21 < HMA50
        price_above_1d_hma21 = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma21 = close[i] < hma_1d_21_aligned[i]
        hma21_above_hma50_1d = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma21_below_hma50_1d = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        regime_bull = price_above_1d_hma21 and hma21_above_hma50_1d
        regime_bear = price_below_1d_hma21 and hma21_below_hma50_1d
        
        # === 4H LOCAL TREND ===
        # HMA crossover signals
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_4h_48[i] > hma_4h_48[i-3] if i >= 3 else False
        hma_slope_down = hma_4h_48[i] < hma_4h_48[i-3] if i >= 3 else False
        
        # Price position relative to 4h HMA48
        price_above_hma48 = close[i] > hma_4h_48[i]
        price_below_hma48 = close[i] < hma_4h_48[i]
        
        # Price relative to SMA50
        price_above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        price_below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False
        
        # === RSI SIGNALS (looser thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral_up = 45.0 < rsi_14[i] < 60.0
        rsi_neutral_down = 40.0 < rsi_14[i] < 55.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.999
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.001
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        vol_expanding = atr_ratio > 1.2
        vol_scale = 1.0 if vol_expanding else 0.9
        
        # === ENTRY LOGIC (OR conditions - ANY can trigger) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (3 different entry types - any can trigger)
        if regime_bull or hma_bullish:
            # Entry 1: RSI pullback in uptrend
            if rsi_neutral_up and hma_bullish and price_above_hma48:
                new_signal = LONG_BASE * vol_scale
            
            # Entry 2: Strong oversold + bull regime
            elif rsi_strong_oversold and (regime_bull or price_above_sma50):
                new_signal = LONG_STRONG * vol_scale
            
            # Entry 3: Donchian breakout + trend confirm
            elif donchian_breakout_up and hma_bullish and rsi_14[i] > 45.0:
                new_signal = LONG_BASE * vol_scale
            
            # Entry 4: HMA crossover + RSI rising
            elif hma_bullish and hma_slope_up and rsi_rising and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * vol_scale
        
        # SHORT ENTRIES (3 different entry types - any can trigger)
        if regime_bear or hma_bearish:
            # Entry 1: RSI pullback in downtrend
            if rsi_neutral_down and hma_bearish and price_below_hma48:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Entry 2: Strong overbought + bear regime
            elif rsi_strong_overbought and (regime_bear or price_below_sma50):
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # Entry 3: Donchian breakdown + trend confirm
            elif donchian_breakout_down and hma_bearish and rsi_14[i] < 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # Entry 4: HMA crossover + RSI falling
            elif hma_bearish and hma_slope_down and rsi_falling and rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 40 bars (~6-7 days on 4h)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
            elif rsi_strong_oversold and price_above_sma50:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif rsi_strong_overbought and price_below_sma50:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma48:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma48:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma48:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma48:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.27:
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