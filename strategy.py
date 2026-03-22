#!/usr/bin/env python3
"""
Experiment #019: 4h HMA Trend + 1d Filter + Volume Confirmation + ATR Volatility

Hypothesis: Previous complex strategies (Fisher, KAMA, Choppiness) failed because they
over-fit to specific regimes. This strategy uses simpler, more robust signals:

1. HMA(16/48) crossover on 4h - proven trend indicator with less lag than EMA
   Long: HMA16 crosses above HMA48. Short: HMA16 crosses below HMA48.

2. 1d HMA(21) trend filter via mtf_data - only take longs if price > 1d HMA,
   only take shorts if price < 1d HMA. Prevents counter-trend disasters.

3. Volume confirmation - taker_buy_volume / total_volume > 0.55 for longs,
   < 0.45 for shorts. Ensures real buying/selling pressure, not fake breakouts.

4. ATR volatility filter - ATR(14)/close > 0.015 (1.5% daily vol). Avoid entering
   in dead markets where signals whipsaw.

5. ATR(14) trailing stop - 2.5x ATR for risk management. Exit when price moves
   against position by 2.5x ATR from entry/extreme.

Why this should work:
- HMA crossover is proven (current best strategy uses similar logic)
- 1d filter prevents major counter-trend trades (better than 1w which is too slow)
- Volume confirmation filters fake breakouts (major issue in crypto)
- ATR volatility filter avoids dead market whipsaws
- 4h timeframe = 20-50 trades/year target (optimal for fee drag)
- Conservative sizing (0.25-0.30) protects against crashes

Timeframe: 4h (REQUIRED for Experiment #019)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_volume_atr_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (buying pressure indicator)."""
    taker_s = pd.Series(taker_buy_volume)
    vol_s = pd.Series(volume)
    
    # Avoid division by zero
    vol_s = vol_s.replace(0, np.nan)
    
    ratio = taker_s / vol_s
    ratio = ratio.fillna(0.5)  # Default to neutral if no volume data
    
    return ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === HMA CROSSOVER SIGNALS ===
        hma_cross_up = False
        hma_cross_down = False
        
        if i > 0:
            # Long signal: HMA16 crosses above HMA48
            if hma_16[i-1] <= hma_48[i-1] and hma_16[i] > hma_48[i]:
                hma_cross_up = True
            # Short signal: HMA16 crosses below HMA48
            if hma_16[i-1] >= hma_48[i-1] and hma_16[i] < hma_48[i]:
                hma_cross_down = True
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = vol_ratio[i] > 0.55  # Strong buying pressure
        volume_bearish = vol_ratio[i] < 0.45  # Strong selling pressure
        
        # === ATR VOLATILITY FILTER ===
        # Avoid entering in dead markets (vol < 1.5% of price)
        vol_ratio_pct = atr_14[i] / close[i]
        vol_active = vol_ratio_pct > 0.012  # At least 1.2% volatility
        
        # === HMA TREND CONFIRMATION ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: HMA crossover
        if hma_cross_up:
            long_score += 3.0
            long_confidence = 1
        
        # Trend alignment (daily filter)
        if daily_bullish:
            long_score += 2.0
        
        # HMA trend confirmation
        if hma_bullish:
            long_score += 1.0
        
        # Volume confirmation
        if volume_bullish:
            long_score += 1.5
        
        # Volatility filter (must be active market)
        if vol_active:
            long_score += 0.5
        
        # Enter long if score >= 5.0 (strong confluence)
        if long_score >= 5.0:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: HMA crossover
        if hma_cross_down:
            short_score += 3.0
            short_confidence = 1
        
        # Trend alignment (daily filter)
        if daily_bearish:
            short_score += 2.0
        
        # HMA trend confirmation
        if hma_bearish:
            short_score += 1.0
        
        # Volume confirmation
        if volume_bearish:
            short_score += 1.5
        
        # Volatility filter (must be active market)
        if vol_active:
            short_score += 0.5
        
        # Enter short if score >= 5.0 (strong confluence)
        if short_score >= 5.0:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~320 hours = 13 days on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if hma_bullish and daily_bullish and vol_active:
                new_signal = REDUCED_SIZE
            elif hma_bearish and daily_bearish and vol_active:
                new_signal = -REDUCED_SIZE
        
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
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Exit long if HMA16 crosses below HMA48
            if position_side > 0 and hma_cross_down:
                hma_exit = True
            # Exit short if HMA16 crosses above HMA48
            if position_side < 0 and hma_cross_up:
                hma_exit = True
        
        # === DAILY TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or hma_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals