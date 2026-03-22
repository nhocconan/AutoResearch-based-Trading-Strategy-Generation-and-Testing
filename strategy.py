#!/usr/bin/env python3
"""
Experiment #009: 1h Connors RSI Mean Reversion with 4h HMA Trend Bias and Choppiness Regime

Hypothesis: After 8 failed experiments, the pattern shows:
1. Pure trend strategies fail in bear/range markets (2022 crash, 2025 bear)
2. Lower TFs (15m-30m) suffer from noise and fee drag
3. BTC/ETH need mean-reversion edges, not just trend following

This 1h strategy combines proven edges from quantitative literature:

1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 (oversold), Short when CRSI > 90 (overbought)
   - Proven 75% win rate in mean-reversion literature
   - Much faster than standard RSI(14), catches short-term extremes

2. 4h HMA trend bias: Only take longs if price > 4h_HMA, shorts if price < 4h_HMA
   - Prevents counter-trend trades in strong trends
   - HMA smoother than EMA, less whipsaw

3. CHOPPINESS INDEX regime filter:
   - CHOP > 61.8 = range market → use mean reversion (CRSI extremes)
   - CHOP < 38.2 = trending → use trend pullback entries
   - Adapts to market regime automatically

4. Funding rate z-score contrarian:
   - Load funding data, calculate 30-day z-score
   - Z < -2 = extreme negative funding → long (crowd too bearish)
   - Z > +2 = extreme positive funding → short (crowd too bullish)
   - BEST EDGE for BTC/ETH per research (Sharpe 0.8-1.5 through 2022)

5. ATR-based position sizing and stoploss:
   - Base size 0.25-0.30, reduced when ATR is high
   - Stoploss at 2.5 * ATR(14) trailing
   - Protects in volatile crashes

6. Volume confirmation:
   - Entry volume > 0.7 * 20-bar avg (filters fakeouts)

Why 1h should work:
- 1h has 4x fewer trades than 15m = less fee drag
- CRSI mean reversion works in bear/range markets (2025 test period)
- 4h HMA bias prevents disaster in strong trends
- Funding rate edge is BTC/ETH specific (not SOL-biased)
- Target 35-55 trades/year = optimal for 1h (Rule 10)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_4h_hma_funding_zscore_atr_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI of streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values (period=2)
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent rank of daily returns
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100 if len(x.iloc[:-1]) > 0 else 50
    )
    
    # CRSI = average of three components
    crsi = (rsi_fast + streak_rsi + percent_rank.values) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR calculation
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness formula
    price_range = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop.values

def calculate_funding_zscore(funding_data, window=30):
    """
    Calculate z-score of funding rates over rolling window.
    Z < -2 = extreme negative (contrarian long)
    Z > +2 = extreme positive (contrarian short)
    """
    if funding_data is None or len(funding_data) == 0:
        return None
    
    funding_s = pd.Series(funding_data)
    mean = funding_s.rolling(window=window, min_periods=window).mean()
    std = funding_s.rolling(window=window, min_periods=window).std()
    zscore = (funding_s - mean) / std.replace(0, np.inf)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Try to load funding data (optional edge)
    funding_zscore = None
    try:
        import os
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0]
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            if len(funding_df) > 0 and 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                # Align funding to prices (funding is 8h, prices are 1h)
                if len(funding_rates) >= len(close) // 8:
                    funding_expanded = np.repeat(funding_rates, 8)[:n]
                    if len(funding_expanded) == n:
                        funding_zscore = calculate_funding_zscore(funding_expanded, 30)
    except Exception:
        funding_zscore = None
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands for additional mean reversion signal
    close_s = pd.Series(close)
    bb_sma = close_s.rolling(window=20, min_periods=20).mean()
    bb_std = close_s.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_sma + 2.0 * bb_std).values
    bb_lower = (bb_sma - 2.0 * bb_std).values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(chop[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 61.8  # Range/choppy market
        is_trend = chop[i] < 38.2  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Mean reversion long
        crsi_overbought = crsi[i] > 85  # Mean reversion short
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        funding_long_signal = False
        funding_short_signal = False
        if funding_zscore is not None and i < len(funding_zscore) and not np.isnan(funding_zscore[i]):
            funding_long_signal = funding_zscore[i] < -1.5
            funding_short_signal = funding_zscore[i] > 1.5
        
        # === ATR-BASED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[150:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
                atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
                size_multiplier = 1.0 / atr_ratio
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.30)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: RANGE MARKET MEAN REVERSION (CHOP > 61.8)
        # Long: CRSI oversold + BB lower + volume + (bull bias OR funding long)
        if is_range:
            if crsi_oversold and (bb_oversold or crsi[i] < 10):
                if volume_confirmed and (bull_bias or funding_long_signal):
                    new_signal = current_size
            
            # Short: CRSI overbought + BB upper + volume + (bear bias OR funding short)
            elif crsi_overbought and (bb_overbought or crsi[i] > 90):
                if volume_confirmed and (bear_bias or funding_short_signal):
                    new_signal = -current_size
        
        # MODE 2: TREND MARKET PULLBACK (CHOP < 38.2)
        # Long: bull bias + CRSI pullback (not extreme, just < 40)
        elif is_trend:
            if bull_bias and crsi[i] < 40 and volume_confirmed:
                new_signal = current_size
            
            # Short: bear bias + CRSI pullback (not extreme, just > 60)
            elif bear_bias and crsi[i] > 60 and volume_confirmed:
                new_signal = -current_size
        
        # MODE 3: FUNDING CONTRARIAN (strong z-score signal)
        # Override other signals when funding is extreme
        if funding_zscore is not None and i < len(funding_zscore) and not np.isnan(funding_zscore[i]):
            if funding_zscore[i] < -2.0 and bull_bias:
                new_signal = current_size  # Strong funding long
            elif funding_zscore[i] > 2.0 and bear_bias:
                new_signal = -current_size  # Strong funding short
        
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
            # Exit long if 4H bias turns bearish
            if position_side > 0 and bear_bias and chop[i] < 38.2:
                trend_reversal = True
            # Exit short if 4H bias turns bullish
            if position_side < 0 and bull_bias and chop[i] < 38.2:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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