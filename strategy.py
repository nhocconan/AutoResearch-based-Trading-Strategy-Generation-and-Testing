#!/usr/bin/env python3
"""
Experiment #583: 1d Primary + 1w HTF — Dual Regime (Chop + Connors RSI)

Hypothesis: Based on #577 (Sharpe=0.520) which used 1d + Choppiness + CRSI + 1w,
this strategy refines the approach with PROVEN patterns from literature:

1. Choppiness Index (CHOP) regime detection:
   - CHOP > 61.8 = range market → mean reversion entries (CRSI extremes)
   - CHOP < 38.2 = trending market → trend follow entries (HMA breakout)
   - This regime switch achieved Sharpe +0.923 on ETH in research

2. 1w HMA(21) for MAJOR trend bias (very slow, reliable):
   - Only long when price > 1w HMA (bull regime)
   - Only short when price < 1w HMA (bear regime)
   - Filters out counter-trend trades that destroy Sharpe

3. Connors RSI for mean reversion entries in chop:
   - Long: CRSI < 15 (extreme oversold) + CHOP > 61.8
   - Short: CRSI > 85 (extreme overbought) + CHOP > 61.8
   - Wider bands than #577 to ensure >=30 trades/symbol

4. HMA breakout for trend entries:
   - Long: price crosses above HMA(21) + CHOP < 38.2 + 1w bull
   - Short: price crosses below HMA(21) + CHOP < 38.2 + 1w bear

5. ATR(14) 2.5x trailing stop for all positions
6. Position size: 0.30 discrete (per Rule 4)

Why this might beat Sharpe=0.520:
- Dual regime adapts to market conditions (chop vs trend)
- 1w HTF is MORE reliable than 1d for major trend direction
- CRSI has 75% win rate for mean reversion in literature
- Simpler than #574/#578 which had 0 trades
- 1d TF targets 20-50 trades/year (Rule 10)

Position sizing: 0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_dual_regime_1w_v2"
timeframe = "1d"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - proven 75% win rate for mean reversion.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.values
    
    # PercentRank - where today's return ranks vs last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].dropna()
        if len(window) > 0:
            current_ret = returns.iloc[i]
            rank = (window < current_ret).sum() / len(window)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
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
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover for trend entries
    prev_hma_21 = 0.0
    prev_hma_50 = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(adx_14[i]) if 'adx_14' in dir() else False:
            continue
        if np.isnan(crsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W MAJOR TREND BIAS (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINNESS REGIME DETECTION ===
        chop_high = chop_14[i] > 61.8  # range market
        chop_low = chop_14[i] < 38.2   # trending market
        
        # === CONNORS RSI EXTREMES (for mean reversion in chop) ===
        crsi_extreme_oversold = crsi_14[i] < 15.0
        crsi_extreme_overbought = crsi_14[i] > 85.0
        
        # === HMA CROSSOVER (for trend follow in trend regime) ===
        hma_cross_bull = (close[i] > hma_1d_21[i]) and (prev_hma_21 > 0) and (close[i-1] <= hma_1d_21[i-1])
        hma_cross_bear = (close[i] < hma_1d_21[i]) and (prev_hma_21 > 0) and (close[i-1] >= hma_1d_21[i-1])
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET (mean reversion)
        if chop_high:
            # Long: CRSI extreme oversold + 1w bull bias
            if crsi_extreme_oversold and bull_regime_1w:
                new_signal = POSITION_SIZE
            
            # Short: CRSI extreme overbought + 1w bear bias
            elif crsi_extreme_overbought and bear_regime_1w:
                new_signal = -POSITION_SIZE
        
        # REGIME 2: TRENDING MARKET (trend follow)
        elif chop_low:
            # Long: HMA cross up + 1w bull bias
            if hma_cross_bull and bull_regime_1w:
                new_signal = POSITION_SIZE
            
            # Short: HMA cross down + 1w bear bias
            elif hma_cross_bear and bear_regime_1w:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
        
        # Update previous HMA for crossover detection
        prev_hma_21 = hma_1d_21[i]
        prev_hma_50 = hma_1d_50[i]
    
    return signals