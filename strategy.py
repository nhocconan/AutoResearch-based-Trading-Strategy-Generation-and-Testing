#!/usr/bin/env python3
"""
Experiment #007: 1d Dual-Regime Strategy with Weekly Trend Filter

Hypothesis: Previous 4h/12h strategies failed due to excessive whipsaws in crypto's
noisy lower timeframes. Daily (1d) timeframe naturally filters noise while maintaining
sufficient trade frequency (target: 20-50 trades/year). This strategy uses:

1. HMA(21/63) Crossover - Fast trend signal with reduced lag vs EMA
2. ADX(14) Regime Filter - ADX>20 = trend mode, ADX<20 = range mode
3. 1w HMA(21) Major Bias - Via mtf_data helper for weekly trend alignment
4. Bollinger Band Mean Reversion - For range-bound markets (ADX<20)
5. ATR(14) Trailing Stop - 2.5x ATR for risk management

Why this should work on 1d:
- Daily candles filter intraday noise that killed 4h strategies
- Dual-regime adapts to both trending AND ranging crypto markets
- Weekly HTF filter prevents counter-trend entries during major moves
- Conservative sizing (0.20-0.30) protects against 77% crashes like 2022
- Simpler logic = more trades (avoids 0-trade failure mode)

Timeframe: 1d (REQUIRED for Experiment #007)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_adx_dual_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.replace([np.inf, -np.inf], np.nan)
    
    return adx.values

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.values, lower.values, middle.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_63 = calculate_hma(close, period=63)
    
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    RANGE_SIZE = 0.20
    
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
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_63[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY TREND ===
        daily_bullish = hma_1d_21[i] > hma_1d_63[i]
        daily_bearish = hma_1d_21[i] < hma_1d_63[i]
        
        # === TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Lowered from 25 for more trades
        adx_weak = adx_14[i] < 20
        
        # === HMA CROSSOVER DETECTION ===
        hma_cross_long = False
        hma_cross_short = False
        
        if i > 0:
            # Long: fast HMA crosses above slow HMA
            if hma_1d_21[i] > hma_1d_63[i] and hma_1d_21[i-1] <= hma_1d_63[i-1]:
                hma_cross_long = True
            # Short: fast HMA crosses below slow HMA
            if hma_1d_21[i] < hma_1d_63[i] and hma_1d_21[i-1] >= hma_1d_63[i-1]:
                hma_cross_short = True
        
        # === MEAN REVERSION SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i]
        bb_break_upper = close[i] > bb_upper[i]
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TREND MODE (ADX >= 20)
        if adx_strong:
            # LONG ENTRY - HMA crossover + weekly alignment
            if hma_cross_long and weekly_bullish:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
            elif daily_bullish and weekly_bullish and not in_position:
                # Continuation entry (no crossover but trend aligned)
                new_signal = BASE_SIZE  # 0.25
            
            # SHORT ENTRY - HMA crossover + weekly alignment
            if hma_cross_short and weekly_bearish:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            elif daily_bearish and weekly_bearish and not in_position:
                # Continuation entry
                new_signal = -BASE_SIZE  # -0.25
        
        # RANGE MODE (ADX < 20) - Mean Reversion
        elif adx_weak:
            # Long at lower BB with oversold RSI
            if bb_break_lower and rsi_oversold:
                new_signal = RANGE_SIZE  # 0.20
            # Short at upper BB with overbought RSI
            elif bb_break_upper and rsi_overbought:
                new_signal = -RANGE_SIZE  # -0.20
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # If no trades for 25 bars (~25 days), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if daily_bullish and weekly_bullish:
                new_signal = BASE_SIZE
            elif daily_bearish and weekly_bearish:
                new_signal = -BASE_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HMA turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if HMA turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === RANGE EXIT (for mean reversion trades) ===
        range_exit = False
        if in_position and position_side != 0 and adx_weak:
            # Exit mean reversion at middle band
            if position_side > 0 and close[i] > bb_middle[i]:
                range_exit = True
            if position_side < 0 and close[i] < bb_middle[i]:
                range_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or range_exit:
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