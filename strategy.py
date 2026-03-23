#!/usr/bin/env python3
"""
Experiment #911: 4h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + CRSI

Hypothesis: After 642 failed strategies, combining Ehlers Fisher Transform (reversal capture)
with Choppiness Index regime detection and Connors RSI should work across ALL symbols.

Key insights from research:
1. Fisher Transform (period=9) catches reversals in bear rallies - proven in literature
2. Choppiness Index is BEST meta-filter for bear/range markets (CHOP>61.8=range, <38.2=trend)
3. Connors RSI has 75% win rate for mean reversion
4. 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
5. 1d/1w HMA provides macro trend bias without over-filtering

Critical improvements:
- RELAXED entry thresholds to guarantee 30+ trades per symbol
- Fisher Transform for reversals (works in bear markets where trend-following fails)
- Multiple entry paths (Fisher + CRSI + Donchian) to ensure trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ATR trailing stop (2.5x) for risk management

Why this should beat Sharpe=0.612:
- Fisher Transform excels in 2022 crash and 2025 bear market (where trend-following failed)
- Choppiness regime switch adapts to market conditions
- 1d/1w HTF provides trend bias without being too restrictive
- Multiple confluence paths ensure trades on BTC/ETH/SOL

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_crsi_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother than EMA, less lag than SMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes. Period=9 is standard.
    
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest_hl2 = np.max(hl2)
        lowest_hl2 = np.min(hl2)
        
        if highest_hl2 == lowest_hl2:
            fisher[i] = 0.0
            continue
        
        # Normalize to 0-1 range
        normalized = (hl2[-1] - lowest_hl2) / (highest_hl2 - lowest_hl2)
        normalized = np.clip(normalized, 0.001, 0.999)  # prevent division by zero
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 20 = oversold, CRSI > 80 = overbought
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(np.concatenate([[0], gain])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(np.concatenate([[0], loss])).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # RSI Streak
    streak = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if direction[i-1] == 1 else 1
            direction[i] = 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if direction[i-1] == -1 else -1
            direction[i] = -1
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        up_count = np.sum(streak_vals > 0)
        total = streak_period
        if total > 0:
            streak_rsi[i] = 100 * up_count / total
    
    # Percent Rank
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = 100 * rank
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures ranging vs trending.
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return np.clip(chop, 0, 100)

def calculate_donchian(high, low, period=20):
    """Donchian Channels - breakout levels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO REGIME (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        ranging_regime = chop_4h[i] > 61.8
        trending_regime = chop_4h[i] < 38.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.5 and fisher_signal_4h[i] >= -1.5  # cross above
        fisher_overbought = fisher_4h[i] > 1.5 and fisher_signal_4h[i] <= 1.5  # cross below
        fisher_extreme_low = fisher_4h[i] < -2.0
        fisher_extreme_high = fisher_4h[i] > 2.0
        
        # === CONNORS RSI SIGNALS (relaxed for trade generation) ===
        crsi_oversold = crsi_4h[i] < 25
        crsi_overbought = crsi_4h[i] > 75
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 61.8) — Mean Reversion ===
        if ranging_regime:
            # Long: Fisher oversold reversal OR CRSI oversold
            if fisher_oversold or crsi_oversold:
                if macro_bull or trend_1d_bullish:  # at least one bullish filter
                    desired_signal = BASE_SIZE
                elif fisher_extreme_low or crsi_extreme_oversold:  # extreme alone
                    desired_signal = REDUCED_SIZE
            
            # Short: Fisher overbought reversal OR CRSI overbought
            if fisher_overbought or crsi_overbought:
                if macro_bear or trend_1d_bearish:  # at least one bearish filter
                    desired_signal = -BASE_SIZE
                elif fisher_extreme_high or crsi_extreme_overbought:  # extreme alone
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + Donchian breakout OR Fisher/CRSI pullback
            if macro_bull or trend_1d_bullish:
                if donchian_breakout_long:
                    desired_signal = BASE_SIZE
                elif fisher_oversold or crsi_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Donchian breakout OR Fisher/CRSI rally
            if macro_bear or trend_1d_bearish:
                if donchian_breakout_short:
                    desired_signal = -BASE_SIZE
                elif fisher_overbought or crsi_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: require confluence
            if (fisher_oversold or crsi_oversold) and (macro_bull or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if (fisher_overbought or crsi_overbought) and (macro_bear or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Fallback: extreme Fisher alone (guarantees some trades)
            if fisher_extreme_low and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if fisher_extreme_high and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher/CRSI not overbought
                if (macro_bull or trend_1d_bullish) and fisher_4h[i] < 1.0 and crsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher/CRSI not oversold
                if (macro_bear or trend_1d_bearish) and fisher_4h[i] > -1.0 and crsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both macro + medium trend reverse
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if Fisher extremely overbought
            if fisher_4h[i] > 2.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both macro + medium trend reverse
            if macro_bull and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if Fisher extremely oversold
            if fisher_4h[i] < -2.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals