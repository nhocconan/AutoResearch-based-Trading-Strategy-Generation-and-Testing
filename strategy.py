#!/usr/bin/env python3
"""
Experiment #016: 12h Donchian-HMA Trend Following with 1d Filter

Hypothesis: Previous strategies failed due to over-filtering (0 trades) or 
mean-reversion in trending markets. Research shows Donchian breakouts work 
well on higher timeframes (12h/1d) with proper trend confirmation.

Strategy Logic:
1. Donchian(20) breakout - primary entry signal (price breaks 20-bar high/low)
2. 1d HMA(21) - major trend filter (only long if price > 1d HMA, short if <)
3. ADX(14) > 18 - trend strength filter (avoid choppy breakouts)
4. HMA(21) vs HMA(50) on 12h - intermediate trend confirmation
5. ATR(14)*2.5 trailing stoploss - risk management
6. Position sizing: 0.25-0.30 discrete levels

Why 12h timeframe:
- Targets 20-50 trades/year (optimal for trend following)
- Less noise than 4h/1h, more signals than 1d
- Proven on SOL (Sharpe +0.782 in research with similar setup)

Key improvements over failed strategies:
- LOOSER entry conditions (ADX > 18 not 25, Donchian not CRSI extremes)
- Frequency safeguard (force entry after 100 bars without trade)
- Simpler trend filter (1d HMA only, not 3-TF alignment)
- Scoring system allows partial confluence entries

Timeframe: 12h (REQUIRED for Experiment #016)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_adx_1d_v1"
timeframe = "12h"
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # ADX calculation
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - breakout system.
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2
    
    return upper.values, lower.values, mid.values

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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H INTERMEDIATE TREND ===
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === TREND STRENGTH ===
        adx_strong = adx_14[i] > 18  # Lower threshold for more trades
        adx_very_strong = adx_14[i] > 25
        
        # === DONCHIAN BREAKOUT ===
        # Check if price broke above upper or below lower in last 3 bars
        breakout_long = False
        breakout_short = False
        
        for lookback in range(3):
            if i - lookback > 0:
                if close[i - lookback] > donchian_upper[i - lookback - 1] if i - lookback - 1 >= 0 else False:
                    breakout_long = True
                if close[i - lookback] < donchian_lower[i - lookback - 1] if i - lookback - 1 >= 0 else False:
                    breakout_short = True
        
        # Current price position relative to Donchian
        price_near_upper = close[i] > donchian_upper[i] * 0.98
        price_near_lower = close[i] < donchian_lower[i] * 1.02
        
        # === RSI FILTER ===
        rsi_bullish = rsi_14[i] > 45 and rsi_14[i] < 75  # Not overbought
        rsi_bearish = rsi_14[i] > 25 and rsi_14[i] < 55  # Not oversold
        
        # === ENTRY SCORING SYSTEM ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY SCORING
        long_score = 0.0
        long_confidence = 0
        
        # Donchian breakout (primary trigger)
        if breakout_long or price_near_upper:
            long_score += 2.5
            long_confidence = 1
        elif close[i] > donchian_mid[i]:
            long_score += 1.0
        
        # 1d trend alignment (major filter)
        if daily_bullish:
            long_score += 2.0
        else:
            long_score -= 1.0  # Penalty for counter-trend
        
        # 12h trend alignment
        if hma_12h_bullish:
            long_score += 1.5
        elif hma_12h_bearish:
            long_score -= 0.5
        
        # ADX trend strength
        if adx_very_strong:
            long_score += 1.5
        elif adx_strong:
            long_score += 0.75
        
        # RSI confirmation
        if rsi_bullish:
            long_score += 0.5
        
        # DI confirmation
        if plus_di[i] > minus_di[i]:
            long_score += 0.5
        
        # Enter long if score >= 5.0 (moderate confluence)
        if long_score >= 5.0:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY SCORING
        short_score = 0.0
        short_confidence = 0
        
        # Donchian breakout (primary trigger)
        if breakout_short or price_near_lower:
            short_score += 2.5
            short_confidence = 1
        elif close[i] < donchian_mid[i]:
            short_score += 1.0
        
        # 1d trend alignment (major filter)
        if daily_bearish:
            short_score += 2.0
        else:
            short_score -= 1.0  # Penalty for counter-trend
        
        # 12h trend alignment
        if hma_12h_bearish:
            short_score += 1.5
        elif hma_12h_bullish:
            short_score -= 0.5
        
        # ADX trend strength
        if adx_very_strong:
            short_score += 1.5
        elif adx_strong:
            short_score += 0.75
        
        # RSI confirmation
        if rsi_bearish:
            short_score += 0.5
        
        # DI confirmation
        if minus_di[i] > plus_di[i]:
            short_score += 0.5
        
        # Enter short if score >= 5.0 (moderate confluence)
        if short_score >= 5.0:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~50 days on 12h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if daily_bullish and hma_12h_bullish and close[i] > donchian_mid[i]:
                new_signal = REDUCED_SIZE
            elif daily_bearish and hma_12h_bearish and close[i] < donchian_mid[i]:
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
        
        # === DONCHIAN REVERSAL EXIT ===
        donchian_exit = False
        if in_position and position_side != 0:
            # Exit long if price breaks Donchian lower
            if position_side > 0 and close[i] < donchian_lower[i]:
                donchian_exit = True
            # Exit short if price breaks Donchian upper
            if position_side < 0 and close[i] > donchian_upper[i]:
                donchian_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if major trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if major trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short if RSI oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or donchian_exit or trend_reversal or rsi_exit:
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