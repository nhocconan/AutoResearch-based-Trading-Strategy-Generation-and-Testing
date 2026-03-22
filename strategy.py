#!/usr/bin/env python3
"""
Experiment #007: 1d Donchian Breakout with Weekly HMA Trend Filter

Hypothesis: Previous regime-switching strategies failed due to over-complexity.
Research shows Donchian breakouts with HMA trend filter worked well on SOL (Sharpe +0.782).
This strategy simplifies to:

1. Donchian(20) breakout - price breaks 20-day high/low
2. 1w HMA(21) for major trend bias - only long if price > weekly HMA
3. RSI(14) filter - avoid entries at extreme levels (RSI 30-70 range preferred)
4. ATR(14) stoploss - 2.5x ATR trailing stop
5. 1d timeframe - targets 20-40 trades/year (optimal for trend following)

Why this should work:
- Donchian breakouts capture major trend moves (BTC 2021 rally, 2022 crash)
- Weekly HMA prevents counter-trend trades (major failure mode in 2022)
- RSI filter avoids chasing extended moves
- 1d TF has fewer trades = less fee drag, better for trend following
- Conservative sizing (0.25-0.30) protects against crashes

Timeframe: 1d (REQUIRED for Experiment #007)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_rsi_1w_v1"
timeframe = "1d"
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr.values

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout indicator.
    Upper = highest high over period
    Lower = lowest low over period
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    mid = (upper + lower) / 2
    
    return upper.values, lower.values, mid.values

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging
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
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # 1d HMA for intermediate trend
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === WEEKLY MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        hma_1d_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_1d_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === TREND STRENGTH (ADX) ===
        adx_trending = adx_14[i] > 22  # Trending market
        adx_weak = adx_14[i] < 18  # Weak/ranging
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI FILTER ===
        rsi_neutral = 35 < rsi_14[i] < 65  # Not overextended
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === DI CROSSOVER ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Donchian breakout + weekly trend + RSI filter
        long_score = 0
        long_confidence = 0
        
        # Primary trigger: Donchian breakout
        if donchian_breakout_long:
            long_score += 2.5
            long_confidence = 1
        elif close[i] > donchian_mid[i] and rsi_oversold:
            long_score += 1.5
            long_confidence = 0.7
        
        # Weekly trend alignment (MUST be bullish for long)
        if weekly_bullish:
            long_score += 2.0
        elif not weekly_bearish:  # Neutral weekly
            long_score += 0.5
        
        # 1d trend confirmation
        if hma_1d_bullish:
            long_score += 1.5
        elif di_bullish:
            long_score += 1.0
        
        # ADX trend strength bonus
        if adx_trending:
            long_score += 1.0
        
        # RSI not overbought (avoid chasing)
        if rsi_14[i] < 65:
            long_score += 0.5
        
        # Enter long if score >= 5.5 (strong confluence)
        if long_score >= 5.5:
            new_signal = BASE_SIZE if long_confidence == 1 else REDUCED_SIZE
        
        # SHORT ENTRY: Donchian breakout + weekly trend + RSI filter
        short_score = 0
        short_confidence = 0
        
        # Primary trigger: Donchian breakout
        if donchian_breakout_short:
            short_score += 2.5
            short_confidence = 1
        elif close[i] < donchian_mid[i] and rsi_overbought:
            short_score += 1.5
            short_confidence = 0.7
        
        # Weekly trend alignment (MUST be bearish for short)
        if weekly_bearish:
            short_score += 2.0
        elif not weekly_bullish:  # Neutral weekly
            short_score += 0.5
        
        # 1d trend confirmation
        if hma_1d_bearish:
            short_score += 1.5
        elif di_bearish:
            short_score += 1.0
        
        # ADX trend strength bonus
        if adx_trending:
            short_score += 1.0
        
        # RSI not oversold (avoid chasing)
        if rsi_14[i] > 35:
            short_score += 0.5
        
        # Enter short if score >= 5.5 (strong confluence)
        if short_score >= 5.5:
            new_signal = -BASE_SIZE if short_confidence == 1 else -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and hma_1d_bullish and rsi_oversold:
                new_signal = REDUCED_SIZE
            elif weekly_bearish and hma_1d_bearish and rsi_overbought:
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI goes overbought (momentum exhausted)
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short if RSI goes oversold (momentum exhausted)
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish and hma_1d_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish and hma_1d_bullish:
                trend_reversal = True
        
        # === DONCHIAN MIDLINE EXIT ===
        donchian_exit = False
        if in_position and position_side != 0:
            # Exit long if price falls back below Donchian mid
            if position_side > 0 and close[i] < donchian_mid[i]:
                donchian_exit = True
            # Exit short if price rises back above Donchian mid
            if position_side < 0 and close[i] > donchian_mid[i]:
                donchian_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or rsi_exit or trend_reversal or donchian_exit:
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