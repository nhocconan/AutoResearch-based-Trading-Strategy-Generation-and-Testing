#!/usr/bin/env python3
"""
Experiment #357: 1d Primary + 1w HTF — Vol-Spike Mean Reversion with Trend Filter

Hypothesis: After 356 experiments, clear patterns emerge:
1. Complex dual-regime strategies (#352-356) all FAILED with negative Sharpe
2. Simpler 1d/1w combinations work best (current best: Sharpe=0.435)
3. Vol-spike mean reversion is underexplored — ATR(7)/ATR(30) > 2.0 signals panic
4. After vol spikes, price mean-reverts to Bollinger mid — capture this "vol crush"
5. 1w HMA(21) for major trend bias (long bias in crypto, but allow shorts in bear)
6. Relaxed RSI thresholds (35/65 not 30/70) to ensure sufficient trade frequency
7. Target: 30-50 trades/year on 1d timeframe

Why this might beat current best (Sharpe=0.435):
- Vol-spike entries capture panic bottoms (2022 crash, 2025 bear rallies)
- Mean-reversion works better than trend-follow in bear/range markets
- 1w trend filter prevents counter-trend disasters
- Simpler logic = fewer whipsaws than dual-regime approaches

Position sizing: 0.25-0.30 (discrete levels to minimize fee churn)
Stoploss: 2.5 * ATR trailing stop
Target: 30-50 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_mr_bb_hma1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    zscore_20 = calculate_zscore(close, period=20)
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        
        # === 1W MAJOR TREND REGIME ===
        trend_bull = close[i] > hma_1w_21_aligned[i]
        trend_bear = close[i] < hma_1w_21_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 2.0 = volatility spike (panic/euphoria)
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.8
        vol_normal = atr_ratio < 1.3
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_oversold = close[i] <= bb_lower[i] * 1.002  # at or below lower band
        bb_overbought = close[i] >= bb_upper[i] * 0.998  # at or above upper band
        bb_mid_cross_up = close[i] > bb_mid[i] and close[i-1] <= bb_mid[i-1] if i > 0 else False
        bb_mid_cross_down = close[i] < bb_mid[i] and close[i-1] >= bb_mid[i-1] if i > 0 else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_neutral = 45.0 < rsi_14[i] < 55.0
        
        # === Z-SCORE EXTREMES ===
        zscore_extreme_long = zscore_20[i] < -1.5
        zscore_extreme_short = zscore_20[i] > 1.5
        
        # === ENTRY LOGIC — VOL SPIKE MEAN REVERSION ===
        new_signal = 0.0
        
        # Long entry: vol spike + oversold conditions + bull trend (or neutral)
        if vol_spike and bb_oversold and rsi_oversold:
            if trend_bull:
                new_signal = LONG_STRONG
            elif not trend_bear:
                new_signal = LONG_BASE
            else:
                # Bear trend but extreme oversold — smaller position
                if zscore_extreme_long:
                    new_signal = LONG_BASE * 0.7
        
        # Long entry: z-score extreme + RSI oversold (no vol spike needed)
        elif zscore_extreme_long and rsi_oversold:
            if trend_bull:
                new_signal = LONG_BASE
            elif not trend_bear:
                new_signal = LONG_BASE * 0.7
        
        # Short entry: vol spike + overbought conditions + bear trend
        if vol_spike and bb_overbought and rsi_overbought:
            if trend_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            elif not trend_bull:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            else:
                # Bull trend but extreme overbought — smaller position
                if zscore_extreme_short and new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
        
        # Short entry: z-score extreme + RSI overbought
        elif zscore_extreme_short and rsi_overbought:
            if trend_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            elif not trend_bull:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.7
        
        # === FREQUENCY BOOSTER — ensure minimum trades ===
        # If no signal for 15 bars (~15 days on 1d), look for weaker entries
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            # Weaker long entry
            if trend_bull and rsi_14[i] < 45.0 and close[i] < bb_mid[i]:
                new_signal = LONG_BASE * 0.6
            # Weaker short entry
            elif trend_bear and rsi_14[i] > 55.0 and close[i] > bb_mid[i]:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.6
            # Mean reversion to BB mid
            elif rsi_neutral and bb_mid_cross_up and trend_bull:
                new_signal = LONG_BASE * 0.5
            elif rsi_neutral and bb_mid_cross_down and trend_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.5
        
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
        
        # === MEAN REVERSION EXIT ===
        # Exit long when price returns to BB mid or RSI neutralizes
        mr_exit_long = False
        if in_position and position_side > 0:
            if close[i] > bb_mid[i] and rsi_14[i] > 55.0:
                mr_exit_long = True
            elif rsi_14[i] > 70.0:  # RSI overbought exit
                mr_exit_long = True
        
        # Exit short when price returns to BB mid or RSI neutralizes
        mr_exit_short = False
        if in_position and position_side < 0:
            if close[i] < bb_mid[i] and rsi_14[i] < 45.0:
                mr_exit_short = True
            elif rsi_14[i] < 30.0:  # RSI oversold exit
                mr_exit_short = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bear and close[i] < hma_1w_21_aligned[i] * 0.98:
                trend_reversal = True
            if position_side < 0 and trend_bull and close[i] > hma_1w_21_aligned[i] * 1.02:
                trend_reversal = True
        
        if stoploss_triggered or mr_exit_long or mr_exit_short or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
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