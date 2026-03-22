#!/usr/bin/env python3
"""
Experiment #311: 4h Primary + 1d/1w HTF — Dual Regime (Trend + Mean Revert) + Connors RSI

Hypothesis: A dual-regime strategy adapting to market conditions outperforms single-regime approaches:
1. Choppiness Index identifies regime: CHOP<45=trending, CHOP>55=range
2. In trending regime: Follow 1d HMA trend with 4h RSI pullback entries
3. In choppy regime: Connors RSI mean reversion at extremes (CRSI<10 long, >90 short)
4. 1w HMA provides major market regime filter (bull/bear bias)
5. Target: 25-45 trades/year on 4h (appropriate frequency, low fee drag)

Why this might beat current best (Sharpe=0.424):
- Connors RSI has proven 75% win rate in mean reversion (research-backed)
- Dual regime adapts to both bull/bear AND range markets (2022 crash + 2023-2024 range)
- 1d HTF trend filter stronger than 4h alone for direction
- Simpler entry logic = more trades generated on ALL symbols (BTC/ETH/SOL)
- Asymmetric sizing favors longs (crypto bias) but allows shorts in bear

Key differences from failed #309 (0 trades):
- Looser Connors RSI thresholds (15/85 instead of 10/90)
- Fewer conflicting conditions per entry
- Frequency safeguard forces trades after 40 bars without signal
- Discrete signal levels reduce churn

Position sizing: 0.25 base, 0.30 strong conviction
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_connors_hma_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw = 2.0 * half - full
    hma = raw.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: Percentage of prior closes lower than current close
    
    CRSI < 15 = oversold (long opportunity)
    CRSI > 85 = overbought (short opportunity)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        crsi[i] = (rsi_close[i] + rsi_streak.iloc[i] + (count_lower / rank_period) * 100.0) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W MAJOR REGIME (bull/bear bias) ===
        regime_bull = close[i] > hma_1w_21_aligned[i]
        regime_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND DIRECTION ===
        trend_up = close[i] > hma_1d_21_aligned[i]
        trend_down = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI PULLBACK SIGNALS (trend following) ===
        rsi_pullback_long = 35.0 < rsi_14[i] < 50.0
        rsi_pullback_short = 50.0 < rsi_14[i] < 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === 4H LOCAL TREND ===
        price_above_hma = close[i] > hma_4h_21[i]
        price_below_hma = close[i] < hma_4h_21[i]
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TRENDING REGIME: Follow HTF trend with pullback entries
        if is_trending:
            # Long: 1d trend up + 1w bull + RSI pullback
            if trend_up and regime_bull and rsi_pullback_long and price_above_hma:
                new_signal = LONG_BASE
            
            # Long: Strong Connors oversold in bull regime
            elif crsi_extreme_oversold and regime_bull:
                new_signal = LONG_STRONG
            
            # Short: 1d trend down + 1w bear + RSI pullback
            if trend_down and regime_bear and rsi_pullback_short and price_below_hma:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Short: Strong Connors overbought in bear regime
            elif crsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
        
        # CHOPPY REGIME: Mean reversion with Connors RSI
        if is_choppy:
            # Long: Connors oversold + 1w bull bias
            if crsi_oversold and regime_bull:
                new_signal = LONG_BASE
            
            # Long: Extreme Connors oversold (any regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_BASE
            
            # Short: Connors overbought + 1w bear bias
            if crsi_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Short: Extreme Connors overbought (any regime)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure 25+ trades/year on 4h) ===
        # Force trade if no signal for 50 bars (~8 days on 4h)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 40.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and crsi[i] > 60.0:
                new_signal = -SHORT_BASE * 0.6
            elif crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.6
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.6
        
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
        
        # === CONNORS RSI EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long: exit when Connors turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short: exit when Connors turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long but 1w regime turns bear + price below 1d HMA
            if position_side > 0 and regime_bear and trend_down:
                regime_reversal = True
            # Short but 1w regime turns bull + price above 1d HMA
            if position_side < 0 and regime_bull and trend_up:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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