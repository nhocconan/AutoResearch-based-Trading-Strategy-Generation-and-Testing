#!/usr/bin/env python3
"""
Experiment #221: 4h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 220 experiments, the key insight is:
1. 4h timeframe provides optimal trade frequency (20-50/year) with manageable noise
2. Donchian(20) breakouts capture sustained moves across all regimes
3. 1d HMA(21) provides cleaner trend filter than 4h indicators
4. RSI(14) momentum confirmation reduces fakeout breakouts
5. SIMPLE logic with LOOSE conditions = guaranteed 10+ trades/symbol

Key differences from failed strategies:
- NO complex regime switching (causes 0 trades in #210, #212, #215, #218)
- NO Choppiness Index (failed in #219 with Sharpe=-0.503)
- NO Connors RSI (failed in #213 with Sharpe=-0.952)
- Direct entry conditions instead of scoring systems
- Loose RSI thresholds (45-55 instead of 42-58) for more trades

Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d1w_v2"
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # 4h HMA for local trend
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_8 = calculate_hma(close, 8)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.14
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_8[i]):
            continue
        
        # === HTF TREND BIAS (1d and 1w) ===
        # Daily trend determines primary bias
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # Weekly trend for major confirmation
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # Daily HMA slope (trend strength)
        daily_strong_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        daily_strong_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        local_bullish = hma_4h_8[i] > hma_4h_21[i]
        local_bearish = hma_4h_8[i] < hma_4h_21[i]
        price_above_4h_hma = close[i] > hma_4h_21[i]
        price_below_4h_hma = close[i] < hma_4h_21[i]
        
        # === MOMENTUM (RSI) - LOOSE THRESHOLDS FOR TRADE FREQUENCY ===
        rsi_bullish = rsi_14[i] > 48  # Loose threshold
        rsi_bearish = rsi_14[i] < 52  # Loose threshold
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        rsi_extreme_bull = rsi_14[i] > 60
        rsi_extreme_bear = rsi_14[i] < 40
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === ENTRY LOGIC - SIMPLE AND LOOSE FOR TRADE FREQUENCY ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple loose paths
        if not in_position or position_side < 0:
            # Path 1: Breakout + daily bullish + RSI bullish (primary)
            if breakout_long and daily_bullish and rsi_bullish:
                new_signal = BASE_SIZE
            
            # Path 2: Breakout + weekly bullish + RSI confirmation
            elif breakout_long and weekly_bullish and rsi_14[i] > 50:
                new_signal = BASE_SIZE
            
            # Path 3: Daily bullish + local bullish + RSI strong (trend continuation)
            elif daily_bullish and local_bullish and rsi_strong_bull and price_above_4h_hma:
                new_signal = BASE_SIZE
            
            # Path 4: Breakout + daily strong bullish (momentum)
            elif breakout_long and daily_strong_bull:
                new_signal = HALF_SIZE
            
            # Path 5: Weekly + daily bullish + RSI moderate (loose entry)
            elif weekly_bullish and daily_bullish and rsi_14[i] > 52 and bars_since_last_trade > 25:
                new_signal = HALF_SIZE
            
            # Path 6: Simple RSI extreme + price above daily HMA (mean reversion in uptrend)
            elif rsi_extreme_bear and daily_bullish and bars_since_last_trade > 35:
                new_signal = HALF_SIZE
        
        # SHORT ENTRIES
        if not in_position or position_side > 0:
            # Path 1: Breakout + daily bearish + RSI bearish (primary)
            if breakout_short and daily_bearish and rsi_bearish:
                new_signal = -BASE_SIZE
            
            # Path 2: Breakout + weekly bearish + RSI confirmation
            elif breakout_short and weekly_bearish and rsi_14[i] < 50:
                new_signal = -BASE_SIZE
            
            # Path 3: Daily bearish + local bearish + RSI strong (trend continuation)
            elif daily_bearish and local_bearish and rsi_strong_bear and price_below_4h_hma:
                new_signal = -BASE_SIZE
            
            # Path 4: Breakout + daily strong bearish (momentum)
            elif breakout_short and daily_strong_bear:
                new_signal = -HALF_SIZE
            
            # Path 5: Weekly + daily bearish + RSI moderate (loose entry)
            elif weekly_bearish and daily_bearish and rsi_14[i] < 48 and bars_since_last_trade > 25:
                new_signal = -HALF_SIZE
            
            # Path 6: Simple RSI extreme + price below daily HMA (mean reversion in downtrend)
            elif rsi_extreme_bull and daily_bearish and bars_since_last_trade > 35:
                new_signal = -HALF_SIZE
        
        # === FREQUENCY SAFEGUARD - Force trades if none for 60 bars ===
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if daily_bullish and weekly_bullish and rsi_14[i] > 50:
                new_signal = HALF_SIZE
            elif daily_bearish and weekly_bearish and rsi_14[i] < 50:
                new_signal = -HALF_SIZE
            elif daily_bullish and rsi_14[i] > 55:
                new_signal = HALF_SIZE * 0.7
            elif daily_bearish and rsi_14[i] < 45:
                new_signal = -HALF_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long positions
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short positions
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily turns strongly bearish
            if position_side > 0 and daily_strong_bear and price_below_4h_hma:
                trend_reversal = True
            # Short position but daily turns strongly bullish
            if position_side < 0 and daily_strong_bull and price_above_4h_hma:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif abs(new_signal) != abs(BASE_SIZE) and abs(new_signal) != abs(HALF_SIZE):
                # Size adjustment without direction change
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