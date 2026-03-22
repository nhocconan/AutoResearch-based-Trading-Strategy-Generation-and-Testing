#!/usr/bin/env python3
"""
Experiment #028: 4h Regime-Adaptive Strategy with Choppiness Index + 1d/1w HMA + Funding Z-Score

Hypothesis: After 27 experiments, the clearest pattern is:
1. Single-regime strategies fail because markets alternate between trending and ranging
2. 4h timeframe is underutilized - needs regime-adaptive logic to work
3. Choppiness Index (CHOP) is the BEST regime detector from literature
4. Funding rate z-score provides BTC/ETH specific edge (contrarian signal)
5. 1d/1w HMA provides robust trend bias without overfitting

This 4h strategy combines:

1. Choppiness Index (14): 
   - CHOP > 61.8 = ranging/choppy market → use mean reversion (RSI + BB)
   - CHOP < 38.2 = trending market → use breakout (Donchian + ADX)
   - Between = neutral (can trade either with reduced size)

2. 1d HMA + 1w HMA dual trend filter:
   - Only take longs when price > at least one HTF HMA
   - Only take shorts when price < at least one HTF HMA
   - Prevents trading against major trend

3. Funding Rate Z-Score (30-day):
   - Z < -2.0 = extreme negative funding → contrarian long signal
   - Z > +2.0 = extreme positive funding → contrarian short signal
   - Uses data/processed/funding/*.parquet

4. Position Sizing:
   - 0.30 for strong signals (regime + HTF + funding all agree)
   - 0.20 for moderate signals (regime + HTF agree)
   - Max 0.35 absolute

5. ATR Trailing Stop: 2.5*ATR(14) for 4h timeframe

Why this should beat current best (Sharpe=0.137):
- Regime adaptation addresses the #1 failure mode (wrong strategy for market state)
- Funding z-score adds BTC/ETH specific alpha not in previous strategies
- Conservative sizing limits drawdown in 2022-style crashes
- 4h TF balances noise filtering with trade frequency (target 30-60 trades/year)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year on 4h (optimal per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_1w_hma_funding_zscore_atr_v2"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) for regime detection.
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper.values, middle.values, lower.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Try to load funding data (BTC/ETH specific edge)
    funding_zscore = np.zeros(n)
    try:
        # Funding data path pattern: data/processed/funding/{symbol}.parquet
        symbol = prices.get('symbol', 'BTCUSDT') if isinstance(prices, dict) else 'BTCUSDT'
        funding_path = f"data/processed/funding/{symbol}.parquet"
        funding_df = pd.read_parquet(funding_path)
        funding_rates = funding_df['funding_rate'].values
        
        # Calculate z-score
        funding_s = pd.Series(funding_rates)
        mean_30 = funding_s.rolling(window=30, min_periods=30).mean()
        std_30 = funding_s.rolling(window=30, min_periods=30).std()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            zscore = (funding_s - mean_30) / std_30
        
        zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        # Pad or truncate to match prices length
        if len(zscore) < n:
            funding_zscore = np.pad(zscore, (0, n - len(zscore)), constant_values=0)
        else:
            funding_zscore = zscore[:n]
    except:
        # No funding data available - use zeros (strategy still works without it)
        funding_zscore = np.zeros(n)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.30  # All filters agree
    SIZE_MODERATE = 0.20  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_ranging and not is_trending  # 38.2 <= CHOP <= 61.8
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        price_vs_1d = close[i] - hma_1d_aligned[i]
        price_vs_1w = close[i] - hma_1w_aligned[i]
        
        bull_htf = (price_vs_1d > 0) or (price_vs_1w > 0)  # At least one bullish
        bear_htf = (price_vs_1d < 0) or (price_vs_1w < 0)  # At least one bearish
        
        # Strong HTF: both agree
        bull_htf_strong = (price_vs_1d > 0) and (price_vs_1w > 0)
        bear_htf_strong = (price_vs_1d < 0) and (price_vs_1w < 0)
        
        # === FUNDING Z-SCORE (contrarian signal) ===
        funding_extreme_long = funding_zscore[i] < -1.5  # Extreme negative funding → long
        funding_extreme_short = funding_zscore[i] > 1.5  # Extreme positive funding → short
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        signal_strength = 0
        
        if is_trending:
            # TRENDING REGIME: Use Donchian breakout + ADX
            
            # Use previous bar's Donchian levels to avoid look-ahead
            donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
            donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
            
            # LONG: Trending + HTF bull + Donchian breakout + ADX/DI confirmation
            if bull_htf and donchian_breakout_long and adx_14[i] > 18:
                signal_strength = 2  # Base: trend regime + breakout
                
                if bull_htf_strong:
                    signal_strength += 1
                
                if di_bull:
                    signal_strength += 1
                
                if funding_extreme_long:
                    signal_strength += 1
                
                if signal_strength >= 4:
                    new_signal = SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = SIZE_MODERATE
            
            # SHORT: Trending + HTF bear + Donchian breakout + ADX/DI confirmation
            elif bear_htf and donchian_breakout_short and adx_14[i] > 18:
                signal_strength = 2  # Base: trend regime + breakout
                
                if bear_htf_strong:
                    signal_strength += 1
                
                if di_bear:
                    signal_strength += 1
                
                if funding_extreme_short:
                    signal_strength += 1
                
                if signal_strength >= 4:
                    new_signal = -SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = -SIZE_MODERATE
        
        elif is_ranging:
            # RANGING REGIME: Use RSI + Bollinger mean reversion
            
            # LONG: Ranging + HTF bull + RSI oversold + price at/near BB lower
            rsi_oversold = rsi_14[i] < 35
            at_bb_lower = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
            
            if bull_htf and rsi_oversold and at_bb_lower:
                signal_strength = 2  # Base: range regime + MR signals
                
                if bull_htf_strong:
                    signal_strength += 1
                
                if rsi_14[i] < 30:  # Very oversold
                    signal_strength += 1
                
                if funding_extreme_long:
                    signal_strength += 1
                
                if signal_strength >= 4:
                    new_signal = SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = SIZE_MODERATE
            
            # SHORT: Ranging + HTF bear + RSI overbought + price at/near BB upper
            rsi_overbought = rsi_14[i] > 65
            at_bb_upper = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
            
            if bear_htf and rsi_overbought and at_bb_upper:
                signal_strength = 2  # Base: range regime + MR signals
                
                if bear_htf_strong:
                    signal_strength += 1
                
                if rsi_14[i] > 70:  # Very overbought
                    signal_strength += 1
                
                if funding_extreme_short:
                    signal_strength += 1
                
                if signal_strength >= 4:
                    new_signal = -SIZE_STRONG
                elif signal_strength >= 3:
                    new_signal = -SIZE_MODERATE
        
        else:  # is_neutral
            # NEUTRAL REGIME: Can trade either strategy but require stronger confirmation
            # Use previous bar's Donchian levels to avoid look-ahead
            donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
            donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
            
            # LONG: HTF bull + Donchian breakout + funding confirmation (need 4+ signals)
            if bull_htf and donchian_breakout_long:
                signal_strength = 2  # Base: HTF + breakout
                
                if bull_htf_strong:
                    signal_strength += 1
                
                if funding_extreme_long:
                    signal_strength += 1
                
                if di_bull:
                    signal_strength += 1
                
                # Require 4+ for neutral regime (more conservative)
                if signal_strength >= 4:
                    new_signal = SIZE_MODERATE
            
            # SHORT: HTF bear + Donchian breakout + funding confirmation
            elif bear_htf and donchian_breakout_short:
                signal_strength = 2  # Base: HTF + breakout
                
                if bear_htf_strong:
                    signal_strength += 1
                
                if funding_extreme_short:
                    signal_strength += 1
                
                if di_bear:
                    signal_strength += 1
                
                # Require 4+ for neutral regime (more conservative)
                if signal_strength >= 4:
                    new_signal = -SIZE_MODERATE
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend strongly reverses against position
            if position_side > 0 and bear_htf_strong:
                trend_exit = True
            if position_side < 0 and bull_htf_strong:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals