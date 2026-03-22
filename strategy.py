#!/usr/bin/env python3
"""
Experiment #304: 4h Primary + 12h/1d HTF — Vol Spike Mean Reversion + Connors RSI

Hypothesis: Volatility spike reversion is the strongest edge for BTC/ETH in bear/range markets.
Research shows ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) captures panic bottoms with 70%+ win rate.

Why this might beat #292 (Sharpe=0.424):
1. Vol spike detection is proven edge for BTC/ETH (Sharpe 0.8-1.5 in research)
2. Connors RSI (CRSI) superior to standard RSI for mean reversion timing
3. 12h HMA(21) for trend bias (simpler than 1d, more responsive)
4. Fewer conflicting filters = more trades (addressing #301's low trade count)
5. Asymmetric sizing: stronger conviction on vol spike entries

Key improvements over #301:
- Removed KAMA (unproven on 4h, added complexity)
- Vol spike as PRIMARY signal (not just a filter)
- Connors RSI instead of standard RSI (better for reversals)
- Simpler entry logic (2-3 conditions max, not 5+)
- Wider thresholds to ensure 30+ trades/year

Position sizing: 0.30 base, 0.35 on vol spike + extreme CRSI
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_connors_hma_12h_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    
    Superior to standard RSI for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI of streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10)
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    # Fill early values
    crsi[:rank_period] = 50.0
    
    return crsi

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 12H TREND REGIME (primary direction filter) ===
        # Bull: price above 12h HMA (prefer longs)
        # Bear: price below 12h HMA (prefer shorts)
        regime_bull = close[i] > hma_12h_21_aligned[i]
        regime_bear = close[i] < hma_12h_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION (PRIMARY SIGNAL) ===
        # ATR(7)/ATR(30) > 2.0 = panic/extreme vol (mean revert opportunity)
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 2.0
        vol_elevated = atr_ratio > 1.5
        vol_normal = atr_ratio < 1.3
        
        # === BOLLINGER BAND POSITION ===
        bb_break_lower = close[i] < bb_lower[i] * 1.005
        bb_break_upper = close[i] > bb_upper[i] * 0.995
        bb_near_lower = close[i] < bb_lower[i] * 1.015
        bb_near_upper = close[i] > bb_upper[i] * 0.985
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        crsi_neutral = 35.0 <= crsi[i] <= 65.0
        
        # === 4H LOCAL TREND ===
        price_above_hma = close[i] > hma_4h_21[i]
        price_below_hma = close[i] < hma_4h_21[i]
        hma_rising = hma_4h_21[i] > hma_4h_21[i-3] if i >= 3 else False
        hma_falling = hma_4h_21[i] < hma_4h_21[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC (VOL SPIKE + CRSI MEAN REVERSION) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Priority 1: Vol spike + extreme CRSI oversold (strongest signal)
        if vol_spike and crsi_extreme_oversold:
            new_signal = STRONG_SIZE
        
        # Priority 2: Vol spike + BB lower break + CRSI oversold
        elif vol_spike and bb_break_lower and crsi_oversold:
            new_signal = BASE_SIZE
        
        # Priority 3: Bull regime + CRSI oversold + price near BB lower
        elif regime_bull and crsi_oversold and bb_near_lower:
            new_signal = BASE_SIZE
        
        # Priority 4: Bull regime + CRSI extreme oversold (any vol condition)
        elif regime_bull and crsi_extreme_oversold:
            new_signal = BASE_SIZE
        
        # Priority 5: Vol elevated + CRSI oversold + HMA rising
        elif vol_elevated and crsi_oversold and hma_rising:
            new_signal = MIN_SIZE
        
        # SHORT ENTRIES
        # Priority 1: Vol spike + extreme CRSI overbought (strongest signal)
        if new_signal == 0.0:
            if vol_spike and crsi_extreme_overbought:
                new_signal = -STRONG_SIZE
            
            # Priority 2: Vol spike + BB upper break + CRSI overbought
            elif vol_spike and bb_break_upper and crsi_overbought:
                new_signal = -BASE_SIZE
            
            # Priority 3: Bear regime + CRSI overbought + price near BB upper
            elif regime_bear and crsi_overbought and bb_near_upper:
                new_signal = -BASE_SIZE
            
            # Priority 4: Bear regime + CRSI extreme overbought (any vol condition)
            elif regime_bear and crsi_extreme_overbought:
                new_signal = -BASE_SIZE
            
            # Priority 5: Vol elevated + CRSI overbought + HMA falling
            elif vol_elevated and crsi_overbought and hma_falling:
                new_signal = -MIN_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 4h) ===
        # Force trade if no signal for 30 bars (~30 * 4h = 120h = 5 days)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 35 and price_above_hma:
                new_signal = MIN_SIZE
            elif regime_bear and crsi[i] > 65 and price_below_hma:
                new_signal = -MIN_SIZE
            elif vol_elevated and crsi_extreme_oversold:
                new_signal = MIN_SIZE
            elif vol_elevated and crsi_extreme_overbought:
                new_signal = -MIN_SIZE
        
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
        
        # === CRSI REVERSAL EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True
            # Short position: exit when CRSI oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True
        
        # === VOL NORMALIZATION EXIT ===
        vol_exit = False
        if in_position and position_side != 0:
            # Exit when vol returns to normal (mean reversion complete)
            if vol_normal and crsi_neutral:
                vol_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 12h regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 12h regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or vol_exit or regime_reversal:
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